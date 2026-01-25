#!/bin/bash
#
# 从 GCS 下载模型文件并上传到腾讯云 COS
#
# 用法: bash move_gcs_to_cos.sh --ak <SECRET_ID> --sk <SECRET_KEY>
#

set -e

# 解析命令行参数
SECRET_ID=""
SECRET_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ak)
            SECRET_ID="$2"
            shift 2
            ;;
        --sk)
            SECRET_KEY="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash move_gcs_to_cos.sh --ak <SECRET_ID> --sk <SECRET_KEY>"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$SECRET_ID" ] || [ -z "$SECRET_KEY" ]; then
    echo "❌ 错误: 请提供 AK/SK"
    echo "用法: bash move_gcs_to_cos.sh --ak <SECRET_ID> --sk <SECRET_KEY>"
    exit 1
fi

echo "==========================================="
echo "GCS -> COS 模型迁移工具"
echo "==========================================="
echo ""

# GCS 文件 URL
GCS_BASE_URL="https://storage.googleapis.com/lxcpublic/krea-models-base-6b5d204f.tar.gz"
GCS_FP8_URL="https://storage.googleapis.com/lxcpublic/krea-models-fp8-f0c953ce.tar.gz"

# COS 配置
COS_BUCKET="rtcos-1394285684"
COS_REGION="ap-nanjing"
COS_BASE_KEY="models/krea-models-base-6b5d204f.tar.gz"
COS_FP8_KEY="models/krea-models-fp8-f0c953ce.tar.gz"

# 临时下载目录
TEMP_DIR="./tmp/gcs_download"
mkdir -p "$TEMP_DIR"

# 安装 coscmd
echo "📦 检查 coscmd..."
pip install -q coscmd

# 配置 coscmd
echo "🔧 配置 coscmd..."
coscmd config -a "$SECRET_ID" -s "$SECRET_KEY" -b "$COS_BUCKET" -r "$COS_REGION"

# 下载并上传函数
download_and_upload() {
    local url=$1
    local cos_key=$2
    local filename=$(basename "$url")
    local local_file="$TEMP_DIR/$filename"
    
    echo ""
    echo "==========================================="
    echo "处理: $filename"
    echo "==========================================="
    
    # 下载
    if [ -f "$local_file" ]; then
        echo "⏭️  文件已存在，跳过下载: $local_file"
    else
        echo "📥 从 GCS 下载..."
        echo "   URL: $url"
        
        if command -v wget &> /dev/null; then
            wget -O "$local_file" "$url"
        elif command -v curl &> /dev/null; then
            curl -L -o "$local_file" "$url"
        else
            echo "❌ 错误: 需要 wget 或 curl"
            exit 1
        fi
        
        echo "✅ 下载完成"
    fi
    
    # 上传到 COS（使用 coscmd，有进度显示）
    echo ""
    echo "📤 上传到 COS..."
    echo "   本地: $local_file"
    echo "   目标: cos://$COS_BUCKET/$cos_key"
    
    coscmd upload "$local_file" "$cos_key"
    
    echo "✅ 完成: $filename"
}

# 处理基础模型
download_and_upload "$GCS_BASE_URL" "$COS_BASE_KEY"

# 处理 FP8 模型
download_and_upload "$GCS_FP8_URL" "$COS_FP8_KEY"

echo ""
echo "==========================================="
echo "✅ 全部完成!"
echo ""
echo "COS 文件:"
echo "  - cos://$COS_BUCKET/$COS_BASE_KEY"
echo "  - cos://$COS_BUCKET/$COS_FP8_KEY"
echo ""
echo "公开访问 URL:"
echo "  - https://$COS_BUCKET.cos.$COS_REGION.myqcloud.com/$COS_BASE_KEY"
echo "  - https://$COS_BUCKET.cos.$COS_REGION.myqcloud.com/$COS_FP8_KEY"
echo "==========================================="

# 可选：清理临时文件
read -p "是否删除临时下载文件? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    echo "✅ 临时文件已删除"
fi
