#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "KREA Realtime Video - Local GPU"
echo "================================="
echo ""

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
}

# Check Python version
echo "✓ Python: $(python3 --version)"

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

echo ""

if [ "$NEED_INSTALL" = true ]; then
    echo "📦 Installing missing dependencies..."
    echo ""
    
    # Install only what's missing
    if ! check_package torch; then
        echo "  - Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    fi
    
    if ! check_package diffusers; then
        echo "  - Installing Diffusers (from source)..."
        pip install git+https://github.com/huggingface/diffusers.git -q
    fi
    
    if ! check_package transformers; then
        echo "  - Installing transformers and accelerate..."
        pip install transformers accelerate safetensors -q
    fi
    
    if ! check_package fastapi; then
        echo "  - Installing FastAPI and utilities..."
        pip install fastapi uvicorn websockets httpx -q
    fi
    
    if ! check_package cv2; then
        echo "  - Installing OpenCV and image processing..."
        pip install opencv-python pillow numpy -q
    fi
    
    if ! check_package msgpack; then
        echo "  - Installing msgpack..."
        pip install msgpack -q
    fi
    
    echo ""
    echo "✅ Dependencies installed!"
    echo ""
fi

# Show GPU info
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Start the server
echo "🚀 Starting KREA Realtime Video server..."
echo ""
echo "📝 Note: First run will download the model (~14GB)"
echo "    This may take 5-10 minutes depending on your network"
echo ""
echo "🌐 Server will be available at: http://0.0.0.0:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "================================="
echo ""

# Run the server
uvicorn app_local:app --host 0.0.0.0 --port 7860
