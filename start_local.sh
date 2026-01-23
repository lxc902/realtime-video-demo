#!/bin/bash
# KREA Realtime Video - 本地 GPU 版本启动脚本

echo "启动 KREA Realtime Video (本地 GPU 模式)..."
echo ""
echo "首次运行会自动下载模型 (~14GB)，请耐心等待"
echo ""

uvicorn app_local:app --host 0.0.0.0 --port 7860
