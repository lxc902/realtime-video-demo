#!/bin/bash

set -e

echo "==========================================="
echo "下载 KREA 模型"
echo "==========================================="
echo ""

# 配置 - 更新这些 URL 为你的 GCS 备份
GCS_BASE_URL="https://storage.googleapis.com/lxcpublic/krea-models-20260123-222800.tar.gz"
GCS_FP8_URL=""  # FP8 模型 URL，留空则跳过

TARGET_DIR="./tmp/.hf_home/hub"

# 解析参数
DOWNLOAD_FP8=false
for arg in "$@"; do
    case $arg in
        --fp8)
            DOWNLOAD_FP8=true
            ;;
    esac
done

# 下载函数
download_model() {
    local url=$1
    local name=$2
    local temp_file="./tmp/krea-model-temp.tar.gz"
    
    echo "📥 下载 $name..."
    echo "   URL: $url"
    
    local success=false
    
    if command -v wget &> /dev/null; then
        if wget --spider -q "$url" 2>/dev/null; then
            wget -O "$temp_file" "$url" && success=true
        fi
    elif command -v curl &> /dev/null; then
        if curl --head --silent --fail "$url" > /dev/null 2>&1; then
            curl -L -o "$temp_file" "$url" && success=true
        fi
    fi
    
    if [ "$success" = true ]; then
        echo "   ✅ 下载成功"
        echo "   📦 解压中..."
        tar -xzf "$temp_file" -C $TARGET_DIR
        rm -f "$temp_file"
        echo "   ✅ 完成"
        return 0
    else
        echo "   ⚠️  下载失败"
        return 1
    fi
}

# 创建目录
mkdir -p ./tmp
mkdir -p $TARGET_DIR

# 下载基础模型
if [ ! -d "$TARGET_DIR/models--krea--krea-realtime-video" ]; then
    if [ -n "$GCS_BASE_URL" ]; then
        download_model "$GCS_BASE_URL" "基础模型" || echo "   将在运行时从 HuggingFace 下载"
    else
        echo "⚠️  基础模型 GCS URL 未配置，将从 HuggingFace 下载"
    fi
else
    echo "✅ 基础模型已存在，跳过"
fi

echo ""

# 下载 FP8 模型（如果指定 --fp8）
if [ "$DOWNLOAD_FP8" = true ]; then
    if [ ! -d "$TARGET_DIR/models--6chan--krea-realtime-video-fp8" ]; then
        if [ -n "$GCS_FP8_URL" ]; then
            download_model "$GCS_FP8_URL" "FP8 模型" || echo "   将在运行时从 HuggingFace 下载"
        else
            echo "⚠️  FP8 模型 GCS URL 未配置，将从 HuggingFace 下载"
        fi
    else
        echo "✅ FP8 模型已存在，跳过"
    fi
fi

echo ""
echo "✅ 模型检查完成"
