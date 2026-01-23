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

echo "📦 准备打包模型..."
echo ""

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR/models--krea--krea-realtime-video" ]; then
    echo "❌ 错误: KREA 模型目录不存在"
    echo "   请先运行 bash run.sh 下载模型"
    exit 1
fi

# 显示要备份的模型
echo "📁 将备份以下模型:"
du -sh $MODEL_DIR/models--krea--krea-realtime-video
du -sh $MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers
echo ""

# 打包模型（跟随符号链接）
echo "🗜️  正在打包模型文件（这可能需要几分钟）..."
cd $MODEL_DIR
tar -czhf /tmp/$BACKUP_NAME \
    models--krea--krea-realtime-video \
    models--Wan-AI--Wan2.1-T2V-14B-Diffusers

echo "✅ 打包完成！"
echo ""

# 显示打包后的文件大小
PACKAGE_SIZE=$(du -sh /tmp/$BACKUP_NAME | cut -f1)
echo "📊 压缩包大小: $PACKAGE_SIZE"
echo "📝 文件名: $BACKUP_NAME"
echo ""

# 上传到 Google Cloud Storage
echo "☁️  上传到 Google Cloud Storage..."
echo "   目标: $BUCKET/$BACKUP_NAME"
echo ""

if ! command -v gsutil &> /dev/null; then
    echo "❌ 错误: gsutil 未安装"
    echo ""
    echo "请安装 Google Cloud SDK:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
    echo "  gcloud init"
    echo ""
    echo "或使用手动上传:"
    echo "  文件位置: /tmp/$BACKUP_NAME"
    exit 1
fi

# 上传（显示进度）
gsutil -m cp /tmp/$BACKUP_NAME $BUCKET/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 上传成功！"
    echo ""
    echo "📥 下载链接（公开）:"
    echo "   https://storage.googleapis.com/lxcpublic/$BACKUP_NAME"
    echo ""
    echo "🔗 使用方式:"
    echo "   wget https://storage.googleapis.com/lxcpublic/$BACKUP_NAME"
    echo "   tar -xzf $BACKUP_NAME -C ~/.cache/huggingface/hub/"
    echo ""
    
    # 清理临时文件
    echo "🧹 清理临时文件..."
    rm /tmp/$BACKUP_NAME
    
    echo "✅ 完成！"
else
    echo ""
    echo "❌ 上传失败"
    echo "   临时文件保存在: /tmp/$BACKUP_NAME"
    exit 1
fi
