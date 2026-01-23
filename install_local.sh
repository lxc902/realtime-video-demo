#!/bin/bash

# KREA Realtime Video - 本地 GPU 版本安装脚本

echo "=== KREA Realtime Video 本地安装 ==="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python3 --version

# 检查 CUDA
echo "检查 CUDA..."
nvidia-smi

# 安装依赖
echo ""
echo "安装依赖..."
pip install --upgrade pip
pip install git+https://github.com/huggingface/diffusers.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors
pip install opencv-python pillow numpy
pip install msgpack

echo ""
echo "=== 安装完成 ==="
echo ""
echo "注意: 首次运行时会自动下载模型（约 14GB），请确保网络畅通"
echo ""
echo "运行方式:"
echo "  python local_inference.py"
echo ""
