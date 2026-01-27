#!/bin/bash

set -e  # Exit on error

# Parse arguments
INSTALL_FLASH_ATTN=false
QUANTIZATION=""
USE_CHINA_MIRROR=false

for arg in "$@"; do
    case $arg in
        --with-flash-attn)
            INSTALL_FLASH_ATTN=true
            ;;
        --fp8)
            QUANTIZATION="fp8"
            ;;
        --int8)
            QUANTIZATION="int8"
            ;;
        --int4)
            QUANTIZATION="int4"
            ;;
        --cn)
            USE_CHINA_MIRROR=true
            ;;
    esac
done

echo "================================="
echo "KREA Realtime Video v2 - Streaming"
if [ -n "$QUANTIZATION" ]; then
    echo "Quantization: ${QUANTIZATION^^}"
fi
if [ "$USE_CHINA_MIRROR" = true ]; then
    echo "Mirror: China (Aliyun)"
fi
echo "================================="
echo ""

# 设置虚拟环境路径（使用绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/tmp/venv"

# 设置 HuggingFace 缓存目录到本地 tmp（必须使用绝对路径）
export HF_HOME="$SCRIPT_DIR/tmp/.hf_home"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/tmp/.hf_home/hub"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/tmp/.hf_home/transformers"

# 如果使用中国镜像，使用 HuggingFace 镜像站点（不能用 OFFLINE 模式，因为 trust_remote_code 需要下载代码）
if [ "$USE_CHINA_MIRROR" = true ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
fi

# PyTorch CUDA 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

# 检查并创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 创建虚拟环境到 $VENV_DIR ..."
    mkdir -p ./tmp
    python3 -m venv "$VENV_DIR"
    echo "✅ 虚拟环境创建完成"
    echo ""
fi

# 创建 HuggingFace 缓存目录
mkdir -p "$HF_HOME"
echo "📂 HuggingFace 缓存目录: $HF_HOME"

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 验证虚拟环境是否激活成功
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "⚠️  警告: 虚拟环境未正确激活"
fi
echo ""

# 使用虚拟环境的 Python 和 pip
PYTHON="$VENV_DIR/bin/python3"
PIP="$PYTHON -m pip"

# Function to check if a Python package is installed
check_package() {
    $PYTHON -c "import $1" 2>/dev/null
}

# Check Python version
echo "✓ Python: $($PYTHON --version)"
echo "✓ Python 位置: $($PYTHON -c 'import sys; print(sys.executable)')"

# Check CUDA/GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ GPU: $GPU_NAME"

# 检测 GPU 架构（CUDA Compute Capability）
detect_gpu_arch() {
    # 尝试用 Python 检测 CUDA capability
    if check_package torch; then
        CUDA_CAP=$($PYTHON -c "import torch; print(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])" 2>/dev/null || echo "0")
    else
        # PyTorch 未安装，根据 GPU 名称判断
        CUDA_CAP="0"
    fi
    
    # 根据 GPU 名称判断架构（先检测，后面会被 CUDA capability 覆盖）
    # Blackwell 检测（包括 RTX PRO 6000 Blackwell, B100, B200, RTX 50xx）
    if echo "$GPU_NAME" | grep -qi "blackwell"; then
        GPU_ARCH="blackwell"
        CUDA_CAP="120"
    elif echo "$GPU_NAME" | grep -qiE "B100|B200|RTX 50"; then
        GPU_ARCH="blackwell"
        CUDA_CAP="120"
    elif echo "$GPU_NAME" | grep -qiE "Ada|RTX 40|L40|RTX 6000 Ada"; then
        GPU_ARCH="ada"
    elif echo "$GPU_NAME" | grep -qiE "Hopper|H100|H200|H800"; then
        GPU_ARCH="hopper"
    elif echo "$GPU_NAME" | grep -qiE "Ampere|A100|RTX 30|A6000"; then
        GPU_ARCH="ampere"
    else
        GPU_ARCH="unknown"
    fi
    
    # 根据 CUDA capability 确定架构
    if [ "$CUDA_CAP" -ge 120 ] 2>/dev/null; then
        GPU_ARCH="blackwell"
    elif [ "$CUDA_CAP" -ge 89 ] 2>/dev/null; then
        GPU_ARCH="ada"
    elif [ "$CUDA_CAP" -ge 90 ] 2>/dev/null; then
        GPU_ARCH="hopper"
    elif [ "$CUDA_CAP" -ge 80 ] 2>/dev/null; then
        GPU_ARCH="ampere"
    fi
    
    echo "$GPU_ARCH"
}

GPU_ARCH=$(detect_gpu_arch)
echo "✓ GPU 架构: $GPU_ARCH"

# 根据 GPU 架构设置软件版本
if [ "$GPU_ARCH" = "blackwell" ]; then
    echo "⚠️  检测到 Blackwell 架构 GPU，将使用 PyTorch nightly (CUDA 12.8)"
    # Blackwell (sm_120) 需要 CUDA 12.8+，cu126 不够
    # 注意：nightly 版本国内镜像通常没有，必须使用官方源
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"
    if [ "$USE_CHINA_MIRROR" = true ]; then
        echo "   ⚠️  PyTorch nightly 需使用官方源（国内镜像无 nightly）"
    fi
    TORCHAO_VERSION=""  # 使用最新版
    TRANSFORMERS_VERSION=""  # 使用最新版
    USE_NIGHTLY=true
    
    # 检查当前 PyTorch 是否支持 Blackwell
    if check_package torch; then
        TORCH_ARCH_LIST=$($PYTHON -c "import torch; print(' '.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "")
        if ! echo "$TORCH_ARCH_LIST" | grep -q "sm_12"; then
            echo "   当前 PyTorch 不支持 sm_120，需要升级"
            BLACKWELL_NEEDS_UPGRADE=true
        else
            BLACKWELL_NEEDS_UPGRADE=false
        fi
    else
        BLACKWELL_NEEDS_UPGRADE=true
    fi
else
    # Ada, Hopper, Ampere 等使用稳定版
    if [ "$USE_CHINA_MIRROR" = true ]; then
        PYTORCH_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121"
    else
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    fi
    TORCHAO_VERSION="==0.7.0"  # 兼容 PyTorch 2.5.x
    TRANSFORMERS_VERSION="==4.44.0"  # 兼容 torchao 0.7.x
    USE_NIGHTLY=false
    BLACKWELL_NEEDS_UPGRADE=false
fi

# 设置 pip 镜像源
if [ "$USE_CHINA_MIRROR" = true ]; then
    PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
    PIP_INDEX_ARGS="-i $PIP_INDEX_URL --trusted-host mirrors.aliyun.com"
    echo "🇨🇳 使用中国镜像源 (阿里云)"
else
    PIP_INDEX_ARGS=""
fi

echo ""

# Check and install missing dependencies
NEED_INSTALL=false

# Blackwell GPU 且 PyTorch 需要升级
if [ "$BLACKWELL_NEEDS_UPGRADE" = true ]; then
    NEED_INSTALL=true
fi

echo "🔍 Checking dependencies..."

if ! check_package torch; then
    echo "  ❌ PyTorch not found"
    NEED_INSTALL=true
else
    echo "  ✓ PyTorch"
fi

# 检查 diffusers 和 huggingface-hub 版本（0.37.0.dev0 + 0.36.0）
DIFFUSERS_VER=$($PYTHON -c "import diffusers; print(diffusers.__version__)" 2>/dev/null || echo "none")
HF_HUB_VER=$($PYTHON -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "none")

if [[ "$DIFFUSERS_VER" == "0.37"* ]] && [[ "$HF_HUB_VER" == "0.36.0" ]]; then
    echo "  ✓ Diffusers ($DIFFUSERS_VER)"
    echo "  ✓ huggingface-hub ($HF_HUB_VER)"
else
    echo "  ⚠️  Diffusers ($DIFFUSERS_VER) / huggingface-hub ($HF_HUB_VER) - 需要安装"
    NEED_INSTALL=true
fi

if ! check_package fastapi; then
    echo "  ❌ FastAPI not found"
    NEED_INSTALL=true
else
    echo "  ✓ FastAPI"
fi

if ! check_package msgpack; then
    echo "  ❌ msgpack not found"
    NEED_INSTALL=true
else
    echo "  ✓ msgpack"
fi

if ! check_package einops; then
    echo "  ❌ einops not found"
    NEED_INSTALL=true
else
    echo "  ✓ einops"
fi

if ! check_package imageio; then
    echo "  ❌ imageio not found"
    NEED_INSTALL=true
else
    echo "  ✓ imageio"
fi

if ! check_package ftfy; then
    echo "  ❌ ftfy not found"
    NEED_INSTALL=true
else
    echo "  ✓ ftfy"
fi

# 如果使用 INT8/INT4 量化，检查 torchao 和 transformers 版本兼容性
if [ "$QUANTIZATION" = "int8" ] || [ "$QUANTIZATION" = "int4" ]; then
    # 检查 torchao
    if ! check_package torchao; then
        echo "  ❌ torchao not found (required for ${QUANTIZATION^^} quantization)"
        NEED_INSTALL=true
    else
        TORCHAO_VER=$($PYTHON -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "unknown")
        if [ "$USE_NIGHTLY" = true ]; then
            # Blackwell: 任何版本都可以（只要能用）
            echo "  ✓ torchao ($TORCHAO_VER)"
        else
            # 旧架构：需要 0.7.x 配合 PyTorch 2.5.x
            if [[ "$TORCHAO_VER" == 0.7* ]]; then
                echo "  ✓ torchao ($TORCHAO_VER)"
            else
                echo "  ⚠️  torchao ($TORCHAO_VER) - 需要 0.7.x 版本 (兼容 PyTorch 2.5.x)"
                NEED_INSTALL=true
            fi
        fi
    fi
    
    # 检查 transformers 版本
    if check_package transformers; then
        TRANSFORMERS_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
        if [ "$USE_NIGHTLY" = true ]; then
            # Blackwell: 任何版本都可以
            echo "  ✓ transformers ($TRANSFORMERS_VER)"
        else
            # 旧架构：需要 4.44.x 配合 torchao 0.7.x
            if [[ "$TRANSFORMERS_VER" == 4.44* ]] || [[ "$TRANSFORMERS_VER" == 4.43* ]] || [[ "$TRANSFORMERS_VER" == 4.42* ]]; then
                echo "  ✓ transformers ($TRANSFORMERS_VER)"
            else
                echo "  ⚠️  transformers ($TRANSFORMERS_VER) - 需要 4.44.x 版本 (兼容 torchao 0.7.x)"
                NEED_INSTALL=true
            fi
        fi
    fi
fi

echo ""

if [ "$NEED_INSTALL" = true ]; then
    echo "📦 Installing missing dependencies..."
    echo "🔍 调试信息:"
    echo "   which python3: $(which python3)"
    echo "   which pip: $(which pip)"
    echo "   \$PYTHON: $PYTHON"
    echo "   测试 pip 安装位置: $($PYTHON -m pip --version)"
    echo ""
    
    # Install or upgrade PyTorch
    if [ "$USE_NIGHTLY" = true ]; then
        # Blackwell GPU: 检查是否需要升级 PyTorch
        if check_package torch; then
            # 检查已安装的 PyTorch 是否支持 sm_120
            TORCH_ARCH_LIST=$($PYTHON -c "import torch; print(' '.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "")
            if echo "$TORCH_ARCH_LIST" | grep -q "sm_12"; then
                echo "  ✓ PyTorch 已支持 Blackwell (sm_120)"
            else
                echo "  ⚠️  当前 PyTorch 不支持 Blackwell，正在升级到 nightly..."
                echo "      当前支持的架构: $TORCH_ARCH_LIST"
                $PIP install --pre torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL --force-reinstall
            fi
        else
            echo "  - Installing PyTorch nightly (for Blackwell GPU)..."
            
            # 中国镜像：从 COS 下载所有 PyTorch nightly 需要的 NVIDIA 依赖
            if [ "$USE_CHINA_MIRROR" = true ]; then
                COS_WHEELS_URL="https://rtcos-1394285684.cos.ap-nanjing.myqcloud.com/pypi/wheels"
                SPECIAL_WHEELS_DIR="$SCRIPT_DIR/vendor/special_wheels"
                
                # PyTorch 2.11.0.dev20260126+cu128 需要的 NVIDIA 依赖（精确版本匹配）
                SPECIAL_PKGS="cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
cuda_pathfinder-1.2.2-py3-none-any.whl
nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl
nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_cudnn_cu12-9.15.1.9-py3-none-manylinux_2_27_x86_64.whl
nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl
nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl
nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl
nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl
nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
triton-3.6.0+git9844da95-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
                
                WHEEL_COUNT=$(ls -1 "$SPECIAL_WHEELS_DIR"/*.whl 2>/dev/null | wc -l)
                if [ "$WHEEL_COUNT" -lt 18 ]; then
                    echo "  - 从 COS 下载 NVIDIA 依赖 (已有 $WHEEL_COUNT/18)..."
                    mkdir -p "$SPECIAL_WHEELS_DIR"
                    
                    for pkg in $SPECIAL_PKGS; do
                        if [ ! -f "$SPECIAL_WHEELS_DIR/$pkg" ]; then
                            echo "    下载: $pkg"
                            wget -q --show-progress -O "$SPECIAL_WHEELS_DIR/$pkg" "$COS_WHEELS_URL/$pkg" 2>&1 || \
                            curl -L --progress-bar -o "$SPECIAL_WHEELS_DIR/$pkg" "$COS_WHEELS_URL/$pkg" || true
                        fi
                    done
                    echo "  ✓ NVIDIA 依赖下载完成"
                fi
                
                # 从本地安装所有 NVIDIA 依赖
                $PIP install --no-index --find-links="$SPECIAL_WHEELS_DIR" \
                    cuda-bindings cuda-pathfinder nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 \
                    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
                    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
                    nvidia-nvshmem-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-cufile-cu12 \
                    triton -q 2>/dev/null || true
                
                # 从 COS 下载 PyTorch nightly wheels
                COS_PYTORCH_URL="https://rtcos-1394285684.cos.ap-nanjing.myqcloud.com/pypi/pytorch"
                PYTORCH_WHEELS_DIR="$SCRIPT_DIR/vendor/pytorch_wheels"
                
                PYTORCH_PKGS="torch-2.11.0.dev20260126+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
torchaudio-2.11.0.dev20260126+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
torchvision-0.25.0.dev20260126+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
                
                PYTORCH_WHEEL_COUNT=$(ls -1 "$PYTORCH_WHEELS_DIR"/*.whl 2>/dev/null | wc -l)
                if [ "$PYTORCH_WHEEL_COUNT" -lt 3 ]; then
                    echo "  - 从 COS 下载 PyTorch nightly wheels (已有 $PYTORCH_WHEEL_COUNT/3)..."
                    mkdir -p "$PYTORCH_WHEELS_DIR"
                    
                    for pkg in $PYTORCH_PKGS; do
                        if [ ! -f "$PYTORCH_WHEELS_DIR/$pkg" ]; then
                            echo "    下载: $pkg"
                            wget -q --show-progress -O "$PYTORCH_WHEELS_DIR/$pkg" "$COS_PYTORCH_URL/$pkg" 2>&1 || \
                            curl -L --progress-bar -o "$PYTORCH_WHEELS_DIR/$pkg" "$COS_PYTORCH_URL/$pkg" || true
                        fi
                    done
                    echo "  ✓ PyTorch wheels 下载完成"
                fi
            fi
            
            # 安装 PyTorch nightly
            PYTORCH_NIGHTLY_VERSION="2.11.0.dev20260126"
            if [ "$USE_CHINA_MIRROR" = true ] && [ -d "$PYTORCH_WHEELS_DIR" ]; then
                # 中国镜像：先从阿里云安装所有网络依赖
                $PIP install setuptools numpy pillow filelock typing-extensions sympy networkx jinja2 fsspec mpmath markupsafe $PIP_INDEX_ARGS -q
                echo "  - 从本地 wheels 安装 PyTorch nightly..."
                $PIP install --no-index --find-links="$PYTORCH_WHEELS_DIR" --find-links="$SPECIAL_WHEELS_DIR" \
                    torch torchvision torchaudio
            else
                # 非中国镜像：从官方源安装
                $PIP install "torch==${PYTORCH_NIGHTLY_VERSION}+cu128" "torchvision==0.25.0.dev20260126+cu128" "torchaudio==${PYTORCH_NIGHTLY_VERSION}+cu128" --index-url $PYTORCH_INDEX_URL
            fi
        fi
    else
        # 其他 GPU: 使用稳定版
        if ! check_package torch; then
            echo "  - Installing PyTorch with CUDA support..."
            $PIP install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL -q
        fi
    fi
    
    # 简化安装：先装 diffusers，再装其他依赖
    LOCAL_DIFFUSERS="$SCRIPT_DIR/../githubrefs/diffusers"
    
    # Step 1: 安装 diffusers 0.37.0.dev0（commit e8e88ff）
    DIFFUSERS_VER=$($PYTHON -c "import diffusers; print(diffusers.__version__)" 2>/dev/null || echo "none")
    if [[ "$DIFFUSERS_VER" != "0.37"* ]]; then
        if [ -d "$LOCAL_DIFFUSERS" ]; then
            echo "  - Installing Diffusers 0.37.x (from local)..."
            $PIP install "$LOCAL_DIFFUSERS" $PIP_INDEX_ARGS -q
        else
            echo "  - Installing Diffusers (from GitHub @e8e88ff)..."
            $PIP install "git+https://github.com/huggingface/diffusers.git@e8e88ff" -q
        fi
    fi
    
    # Step 2: 固定 huggingface-hub==0.36.0（和 requirements.txt 一致）
    echo "  - Fixing huggingface-hub==0.36.0..."
    $PIP install "huggingface-hub==0.36.0" $PIP_INDEX_ARGS -q
    
    # Step 3: 安装其他依赖
    echo "  - Installing other dependencies..."
    $PIP install transformers==4.57.6 accelerate safetensors $PIP_INDEX_ARGS -q
    $PIP install fastapi uvicorn websockets httpx $PIP_INDEX_ARGS -q
    $PIP install opencv-python pillow numpy $PIP_INDEX_ARGS -q
    $PIP install msgpack $PIP_INDEX_ARGS -q
    # 从本地 wheels 安装（避免网络问题）
    if [ -d "$SCRIPT_DIR/vendor/wheels" ]; then
        $PIP install --no-index --find-links="$SCRIPT_DIR/vendor/wheels" einops imageio ftfy protobuf -q
    else
        $PIP install einops imageio ftfy protobuf $PIP_INDEX_ARGS -q
    fi
    
    # Step 4: 再次固定 huggingface-hub（防止被覆盖）
    $PIP install "huggingface-hub==0.36.0" $PIP_INDEX_ARGS -q
    
    # Optional: flash-attention for better performance (disabled by default)
    if [ "$INSTALL_FLASH_ATTN" = true ]; then
        echo "  - Installing flash-attention (for better performance, ~5-10 min)..."
        $PIP install flash-attn --no-build-isolation $PIP_INDEX_ARGS 2>&1 | grep -E "(Installing|Successfully|error)" || echo "    (flash-attn install failed, will use standard attention)"
    else
        echo "  - Skipping flash-attention (use --with-flash-attn to install)"
    fi
    
    # INT8/INT4 量化依赖安装
    if [ "$QUANTIZATION" = "int8" ] || [ "$QUANTIZATION" = "int4" ]; then
        echo "  - 配置 ${QUANTIZATION^^} 量化依赖..."
        
        if [ "$USE_NIGHTLY" = true ]; then
            # Blackwell 使用 torchao（不动 transformers）
            echo "    Blackwell GPU: 安装 torchao..."
            $PIP install torchao $PIP_INDEX_ARGS -q
        else
            # 旧架构使用 torchao 0.7.x
            TORCHAO_VER=$($PYTHON -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "none")
            if [[ "$TORCHAO_VER" != 0.7* ]]; then
                echo "    安装 torchao${TORCHAO_VERSION} (兼容 PyTorch 2.5.x)..."
                $PIP install torchao${TORCHAO_VERSION} $PIP_INDEX_ARGS -q
            fi
        fi
        # 再次固定 huggingface-hub
        $PIP install "huggingface-hub==0.36.0" $PIP_INDEX_ARGS -q
        
        echo "    ✅ ${QUANTIZATION^^} 量化依赖已配置"
    fi
    
    # 最终验证关键版本
    echo ""
    echo "🔍 最终版本验证..."
    FINAL_DIFFUSERS=$($PYTHON -c "import diffusers; print(diffusers.__version__)" 2>/dev/null || echo "none")
    FINAL_HF_HUB=$($PYTHON -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "none")
    
    echo "  ✓ diffusers: $FINAL_DIFFUSERS"
    echo "  ✓ huggingface-hub: $FINAL_HF_HUB"
    
    echo ""
    echo "✅ Dependencies installed!"
    echo "📊 安装位置: $($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
    echo ""
    
    # 验证关键包是否可以正常导入
    echo "🔍 验证安装..."
    if $PYTHON -c "import torch, diffusers, fastapi" 2>/dev/null; then
        echo "✓ 所有包导入正常"
    else
        echo "⚠️  导入失败，请检查错误信息"
    fi
    echo ""
fi

# Show GPU info
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# 预下载模型（--cn 从 COS 下载，否则从 GCS 下载，失败则从 HuggingFace 下载）
echo "📦 检查模型..."
DOWNLOAD_ARGS=""
if [ "$USE_CHINA_MIRROR" = true ]; then
    DOWNLOAD_ARGS="$DOWNLOAD_ARGS --cn"
fi
if [ "$QUANTIZATION" = "fp8" ]; then
    DOWNLOAD_ARGS="$DOWNLOAD_ARGS --fp8"
fi
bash download.sh $DOWNLOAD_ARGS
echo ""

# 设置量化环境变量
if [ -n "$QUANTIZATION" ]; then
    export QUANTIZATION="$QUANTIZATION"
    echo "🔧 量化模式: ${QUANTIZATION^^}"
    echo ""
fi

# Start the server
echo "🚀 Starting KREA Realtime Video server..."
echo ""
echo "📝 Note: 模型下载约 14GB，加载后需要 ~47GB 显存"
echo "    (优先从 GCS 下载，失败则从 HuggingFace 下载)"
echo ""
if [ -n "$QUANTIZATION" ]; then
    echo "💾 使用 ${QUANTIZATION^^} 量化 (显存占用更少)"
else
    echo "⚠️  未启用量化，需要 ~54GB+ 显存"
    echo "   如果 OOM，请使用: bash run.sh --int8 或 --int4"
fi
echo ""
echo "🌐 Server will be available at: http://0.0.0.0:6006"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "================================="
echo ""

# Run the server
$PYTHON -m uvicorn app:app --host 0.0.0.0 --port 6006
