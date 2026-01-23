#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "KREA Realtime Video - Local GPU"
echo "================================="
echo ""

# Check if running first time (check for key dependencies)
if ! python3 -c "import torch" 2>/dev/null; then
    echo "📦 First run detected - installing dependencies..."
    echo ""
    
    # Check Python version
    echo "✓ Checking Python version..."
    python3 --version
    
    # Check CUDA
    echo "✓ Checking CUDA/GPU..."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    echo ""
    echo "📥 Installing Python packages (this may take 5-10 minutes)..."
    
    # Upgrade pip
    pip install --upgrade pip -q
    
    # Install core dependencies
    echo "  - Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    
    echo "  - Installing Diffusers (from source)..."
    pip install git+https://github.com/huggingface/diffusers.git -q
    
    echo "  - Installing transformers and accelerate..."
    pip install transformers accelerate safetensors -q
    
    echo "  - Installing FastAPI and utilities..."
    pip install fastapi uvicorn websockets httpx -q
    pip install opencv-python pillow numpy msgpack -q
    
    echo ""
    echo "✅ Dependencies installed successfully!"
    echo ""
else
    echo "✓ Dependencies already installed"
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
