#!/bin/bash

set -e

echo "==========================================="
echo "下载 KREA 模型"
echo "==========================================="
echo ""

# 配置
GCS_URL="https://storage.googleapis.com/lxcpublic/krea-models-20260123-222800.tar.gz"
TARGET_DIR="./tmp/.hf_home/hub"

# 检查模型是否已存在
if [ -d "$TARGET_DIR/models--krea--krea-realtime-video" ]; then
    echo "✅ 模型已存在，跳过下载"
    exit 0
fi

echo "📥 尝试从 Google Cloud Storage 下载..."
echo "   URL: $GCS_URL"
echo ""

# 创建目录
mkdir -p ./tmp
mkdir -p $TARGET_DIR

# 尝试从 GCS 下载
GCS_SUCCESS=false

# 使用 wget 或 curl 下载
if command -v wget &> /dev/null; then
    echo "   使用 wget 下载..."
    if wget --spider -q "$GCS_URL" 2>/dev/null; then
        wget -O ./tmp/krea-models.tar.gz "$GCS_URL" && GCS_SUCCESS=true
    fi
elif command -v curl &> /dev/null; then
    echo "   使用 curl 下载..."
    if curl --head --silent --fail "$GCS_URL" > /dev/null 2>&1; then
        curl -L -o ./tmp/krea-models.tar.gz "$GCS_URL" && GCS_SUCCESS=true
    fi
fi

if [ "$GCS_SUCCESS" = true ]; then
    echo ""
    echo "✅ GCS 下载成功"
    echo ""
    echo "📦 解压模型..."
    tar -xzf ./tmp/krea-models.tar.gz -C $TARGET_DIR
    
    echo ""
    echo "🧹 清理临时文件..."
    rm -f ./tmp/krea-models.tar.gz
    
    echo ""
    echo "✅ 模型下载完成！"
else
    echo ""
    echo "⚠️  GCS 下载失败或不可用"
    echo "   将在运行时从 HuggingFace 自动下载"
    echo ""
fi
