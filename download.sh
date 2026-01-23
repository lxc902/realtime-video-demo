#!/bin/bash

set -e

echo "==========================================="
echo "从 Google Cloud Storage 下载 KREA 模型"
echo "==========================================="
echo ""

# 配置
BUCKET_URL="https://storage.googleapis.com/lxcpublic"
fTARGET_DIR="./tmp/.hf_home/hub"  # 使用本地 tmp 目录

# 显示可用的备份文件
echo "📋 可用的备份文件:"
echo ""
echo "请访问查看所有备份:"
echo "  https://storage.googleapis.com/lxcpublic/"
echo ""

# 提示用户输入文件名
read -p "请输入要下载的文件名（例如: krea-models-20260123-210000.tar.gz）: " BACKUP_NAME

if [ -z "$BACKUP_NAME" ]; then
    echo "❌ 错误: 未输入文件名"
    exit 1
fi

DOWNLOAD_URL="$BUCKET_URL/$BACKUP_NAME"

echo ""
echo "📥 下载模型..."
echo "   从: $DOWNLOAD_URL"
echo ""

# 下载到临时目录（使用项目本地目录）
mkdir -p ./tmp
wget -O ./tmp/$BACKUP_NAME $DOWNLOAD_URL

if [ $? -ne 0 ]; then
    echo "❌ 下载失败"
    exit 1
fi

echo ""
echo "✅ 下载完成"
echo ""

# 解压
echo "📦 解压模型到 $TARGET_DIR ..."
mkdir -p $TARGET_DIR
tar -xzf ./tmp/$BACKUP_NAME -C $TARGET_DIR

echo ""
echo "✅ 解压完成！"
echo ""

# 清理
echo "🧹 清理临时文件..."
rm ./tmp/$BACKUP_NAME

echo ""
echo "✅ 模型恢复完成！"
echo ""
echo "现在可以运行: bash run.sh"
