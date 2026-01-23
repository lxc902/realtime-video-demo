#!/bin/bash

set -e

echo "==========================================="
echo "KREA 模型备份到 Google Cloud Storage"
echo "==========================================="
echo ""

# 配置
BUCKET="gs://lxcpublic"
MODEL_DIR="/workspace/.hf_home/hub"
BACKUP_NAME="krea-models-$(date +%Y%m%d-%H%M%S).tar.gz"
TEMP_DIR="/workspace"  # 使用 /workspace 而不是 /tmp，空间更大

echo "📦 准备打包模型..."
echo ""

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR/models--krea--krea-realtime-video" ]; then
    echo "❌ 错误: KREA 模型目录不存在"
    echo "   请先运行 bash run.sh 下载模型"
    exit 1
fi

# 检查 /workspace 空间
AVAILABLE_SPACE=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
echo "📊 /workspace 可用空间: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 25 ]; then
    echo "⚠️  警告: 空间可能不足，建议至少 25GB"
    echo ""
    read -p "是否继续? [y/N]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 显示要备份的模型
echo "📁 将备份以下模型:"
du -sh $MODEL_DIR/models--krea--krea-realtime-video
du -sh $MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers
echo ""

# 检查是否安装了 gsutil
if ! command -v gsutil &> /dev/null; then
    echo "❌ 错误: gsutil 未安装"
    echo ""
    echo "请安装 Google Cloud SDK:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
    echo "  gcloud init"
    echo ""
    exit 1
fi

echo "🗜️  正在打包并直接上传（不占用本地空间）..."
echo "   目标: $BUCKET/$BACKUP_NAME"
echo ""

# 直接流式上传，不创建临时文件
cd $MODEL_DIR
tar -czf - \
    models--krea--krea-realtime-video \
    models--Wan-AI--Wan2.1-T2V-14B-Diffusers \
    | gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
             cp - $BUCKET/$BACKUP_NAME

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 上传成功！"
    echo ""
    echo "📥 下载链接（公开）:"
    echo "   https://storage.googleapis.com/lxcpublic/$BACKUP_NAME"
    echo ""
    echo "🔗 使用方式:"
    echo "   wget https://storage.googleapis.com/lxcpublic/$BACKUP_NAME"
    echo "   tar -xzf $BACKUP_NAME -C /workspace/.hf_home/hub/"
    echo ""
    echo "✅ 完成！"
else
    echo ""
    echo "❌ 上传失败"
    exit 1
fi
