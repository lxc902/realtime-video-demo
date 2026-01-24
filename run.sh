#!/bin/bash

set -e  # Exit on error

# Parse arguments
INSTALL_FLASH_ATTN=false
QUANTIZATION=""

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
    esac
done

echo "================================="
echo "KREA Realtime Video - Local GPU"
if [ -n "$QUANTIZATION" ]; then
    echo "Quantization: ${QUANTIZATION^^}"
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
echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Check and install missing dependencies
NEED_INSTALL=false

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
        # 检查 torchao 版本是否兼容（需要 0.7.x 配合 PyTorch 2.5.x）
        TORCHAO_VER=$($PYTHON -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$TORCHAO_VER" == 0.7* ]]; then
            echo "  ✓ torchao ($TORCHAO_VER)"
        else
            echo "  ⚠️  torchao ($TORCHAO_VER) - 需要 0.7.x 版本"
            NEED_INSTALL=true
        fi
    fi
    
    # 检查 transformers 版本（需要 4.44.x 配合 torchao 0.7.x）
    if check_package transformers; then
        TRANSFORMERS_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$TRANSFORMERS_VER" == 4.44* ]] || [[ "$TRANSFORMERS_VER" == 4.43* ]] || [[ "$TRANSFORMERS_VER" == 4.42* ]]; then
            echo "  ✓ transformers ($TRANSFORMERS_VER)"
        else
            echo "  ⚠️  transformers ($TRANSFORMERS_VER) - 需要 4.44.x 版本 (兼容 torchao 0.7.x)"
            NEED_INSTALL=true
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
    
    # Install only what's missing
    if ! check_package torch; then
        echo "  - Installing PyTorch with CUDA support..."
        $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    fi
    
    if ! check_package diffusers; then
        echo "  - Installing Diffusers (from source)..."
        # 先尝试安装最新版本
        $PIP install git+https://github.com/huggingface/diffusers.git -q
        
        # 验证安装，如果失败则尝试稳定版本
        if ! $PYTHON -c "import diffusers" 2>/dev/null; then
            echo "    ⚠️  最新版本安装失败，尝试稳定版本..."
            $PIP install --force-reinstall "diffusers>=0.32.0" -q
        fi
    fi
    
    if ! check_package transformers; then
        echo "  - Installing transformers and accelerate..."
        # INT8/INT4 量化需要 transformers 4.44.x（兼容 torchao 0.7.x）
        if [ "$QUANTIZATION" = "int8" ] || [ "$QUANTIZATION" = "int4" ]; then
            $PIP install transformers==4.44.0 accelerate safetensors -q
        else
            $PIP install transformers accelerate safetensors -q
        fi
    fi
    
    if ! check_package fastapi; then
        echo "  - Installing FastAPI and utilities..."
        $PIP install fastapi uvicorn websockets httpx -q
    fi
    
    if ! check_package cv2; then
        echo "  - Installing OpenCV and image processing..."
        $PIP install opencv-python pillow numpy -q
    fi
    
    if ! check_package msgpack; then
        echo "  - Installing msgpack..."
        $PIP install msgpack -q
    fi
    
    if ! check_package einops; then
        echo "  - Installing einops..."
        $PIP install einops -q
    fi
    
    if ! check_package imageio; then
        echo "  - Installing imageio..."
        $PIP install imageio -q
    fi
    
    if ! check_package ftfy; then
        echo "  - Installing ftfy..."
        $PIP install ftfy -q
    fi
    
    # Optional: flash-attention for better performance (disabled by default)
    if [ "$INSTALL_FLASH_ATTN" = true ]; then
        echo "  - Installing flash-attention (for better performance, ~5-10 min)..."
        $PIP install flash-attn --no-build-isolation 2>&1 | grep -E "(Installing|Successfully|error)" || echo "    (flash-attn install failed, will use standard attention)"
    else
        echo "  - Skipping flash-attention (use --with-flash-attn to install)"
    fi
    
    # INT8/INT4 量化需要 torchao 0.7.x + transformers 4.44.x（兼容 PyTorch 2.5.x）
    if [ "$QUANTIZATION" = "int8" ] || [ "$QUANTIZATION" = "int4" ]; then
        echo "  - 配置 ${QUANTIZATION^^} 量化依赖..."
        
        # 检查并安装 torchao 0.7.x
        TORCHAO_VER=$($PYTHON -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "none")
        if [[ "$TORCHAO_VER" != 0.7* ]]; then
            echo "    安装 torchao==0.7.0 (兼容 PyTorch 2.5.x)..."
            $PIP install torchao==0.7.0 -q
        fi
        
        # 检查并安装 transformers 4.44.x（兼容 torchao 0.7.x）
        TRANSFORMERS_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
        if [[ "$TRANSFORMERS_VER" != 4.44* ]] && [[ "$TRANSFORMERS_VER" != 4.43* ]] && [[ "$TRANSFORMERS_VER" != 4.42* ]]; then
            echo "    安装 transformers==4.44.0 (兼容 torchao 0.7.x)..."
            $PIP install transformers==4.44.0 -q
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
        $PIP install --force-reinstall git+https://github.com/huggingface/diffusers.git -q || \
        $PIP install --force-reinstall "diffusers>=0.32.0" -q
    else
        echo "✓ 所有包导入正常"
    fi
    echo ""
fi

# Show GPU info
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# 预下载模型（优先从 GCS，失败则让 HuggingFace 自动下载）
echo "📦 检查模型..."
bash download.sh
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
