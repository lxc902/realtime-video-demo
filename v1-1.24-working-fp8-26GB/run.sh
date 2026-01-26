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
echo "KREA Realtime Video - Local GPU"
if [ -n "$QUANTIZATION" ]; then
    echo "Quantization: ${QUANTIZATION^^}"
fi
if [ "$USE_CHINA_MIRROR" = true ]; then
    echo "Mirror: China (Tsinghua)"
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

# 如果使用中国镜像，启用离线模式（避免连接 HuggingFace）
if [ "$USE_CHINA_MIRROR" = true ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
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
    elif echo "$GPU_NAME" | grep -qiE "Hopper|H100|H200"; then
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
    PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    PIP_INDEX_ARGS="-i $PIP_INDEX_URL --trusted-host pypi.tuna.tsinghua.edu.cn"
    echo "🇨🇳 使用中国镜像源 (清华)"
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

if ! check_package diffusers; then
        echo "  ❌ Diffusers not found"
        NEED_INSTALL=true
    else
        echo "  ✓ Diffusers"
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
            $PIP install --pre torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
        fi
    else
        # 其他 GPU: 使用稳定版
        if ! check_package torch; then
            echo "  - Installing PyTorch with CUDA support..."
            $PIP install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL -q
        fi
    fi
    
    # 检查是否需要安装/更新 diffusers
    # 注意：模型需要 diffusers 0.36.0.dev0，使用 modular_blocks.py 中的自定义 WanRTBlocks
    # 不需要检查 WanRTBlocks 是否在 diffusers 中，因为它是模型自定义代码
    if ! check_package diffusers; then
        if [ "$USE_CHINA_MIRROR" = true ]; then
            echo "  - Installing Diffusers..."
            $PIP install "diffusers>=0.36.0" $PIP_INDEX_ARGS -q
        else
            echo "  - Installing Diffusers (from source)..."
            $PIP install git+https://github.com/huggingface/diffusers.git -q
        fi
    fi
    
    if ! check_package transformers; then
        echo "  - Installing transformers and accelerate..."
        # 根据 GPU 架构和量化模式选择版本
        if [ "$USE_NIGHTLY" = true ]; then
            # Blackwell 使用最新版
            $PIP install transformers accelerate safetensors $PIP_INDEX_ARGS -q
        elif [ "$QUANTIZATION" = "int8" ] || [ "$QUANTIZATION" = "int4" ]; then
            # 旧架构 + 量化需要 transformers 4.44.x（兼容 torchao 0.7.x）
            $PIP install transformers${TRANSFORMERS_VERSION} accelerate safetensors $PIP_INDEX_ARGS -q
        else
            $PIP install transformers accelerate safetensors $PIP_INDEX_ARGS -q
        fi
    fi
    
    if ! check_package fastapi; then
        echo "  - Installing FastAPI and utilities..."
        $PIP install fastapi uvicorn websockets httpx $PIP_INDEX_ARGS -q
    fi
    
    if ! check_package cv2; then
        echo "  - Installing OpenCV and image processing..."
        $PIP install opencv-python pillow numpy $PIP_INDEX_ARGS -q
    fi
    
    if ! check_package msgpack; then
        echo "  - Installing msgpack..."
        $PIP install msgpack $PIP_INDEX_ARGS -q
    fi
    
    if ! check_package einops; then
        echo "  - Installing einops..."
        $PIP install einops $PIP_INDEX_ARGS -q
    fi
    
    if ! check_package imageio; then
        echo "  - Installing imageio..."
        $PIP install imageio $PIP_INDEX_ARGS -q
    fi
    
    if ! check_package ftfy; then
        echo "  - Installing ftfy..."
        $PIP install ftfy $PIP_INDEX_ARGS -q
    fi
    
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
            # Blackwell 使用最新版 torchao 和 transformers
            echo "    Blackwell GPU: 升级到最新版 torchao 和 transformers..."
            $PIP install --upgrade torchao transformers $PIP_INDEX_ARGS
        else
            # 旧架构使用 torchao 0.7.x + transformers 4.44.x
            TORCHAO_VER=$($PYTHON -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "none")
            if [[ "$TORCHAO_VER" != 0.7* ]]; then
                echo "    安装 torchao${TORCHAO_VERSION} (兼容 PyTorch 2.5.x)..."
                $PIP install torchao${TORCHAO_VERSION} $PIP_INDEX_ARGS -q
            fi
            
            # 检查并安装 transformers 4.44.x（兼容 torchao 0.7.x）
            TRANSFORMERS_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
            if [[ "$TRANSFORMERS_VER" != 4.44* ]] && [[ "$TRANSFORMERS_VER" != 4.43* ]] && [[ "$TRANSFORMERS_VER" != 4.42* ]]; then
                echo "    安装 transformers${TRANSFORMERS_VERSION} (兼容 torchao 0.7.x)..."
                $PIP install transformers${TRANSFORMERS_VERSION} $PIP_INDEX_ARGS -q
            fi
        fi
        
        echo "    ✅ ${QUANTIZATION^^} 量化依赖已配置"
    fi
    
    echo ""
    echo "✅ Dependencies installed!"
    echo "📊 安装位置: $($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
    echo ""
    
    # 验证关键包是否可以正常导入
    echo "🔍 验证安装..."
    if ! $PYTHON -c "import torch, diffusers, fastapi" 2>/dev/null; then
        echo "⚠️  检测到导入问题，尝试修复..."
        echo "   重新安装 diffusers..."
        $PIP install --force-reinstall "diffusers>=0.32.0" $PIP_INDEX_ARGS -q
    else
        echo "✓ 所有包导入正常"
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
echo "🌐 Server will be available at: http://0.0.0.0:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "================================="
echo ""

# Run the server
$PYTHON -m uvicorn app_local:app --host 0.0.0.0 --port 7860
