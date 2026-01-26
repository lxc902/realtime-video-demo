#!/bin/bash

set -e

echo "==========================================="
echo "下载 KREA 模型"
echo "==========================================="
echo ""

# 解析参数
DOWNLOAD_FP8=false
USE_CHINA_MIRROR=false

for arg in "$@"; do
    case $arg in
        --fp8)
            DOWNLOAD_FP8=true
            ;;
        --cn)
            USE_CHINA_MIRROR=true
            ;;
    esac
done

# 配置下载源
if [ "$USE_CHINA_MIRROR" = true ]; then
    # 腾讯云 COS（中国源）
    BASE_URL="https://rtcos-1394285684.cos.ap-nanjing.myqcloud.com/models/krea-models-base-6b5d204f.tar.gz"
    FP8_URL="https://rtcos-1394285684.cos.ap-nanjing.myqcloud.com/models/krea-models-fp8-f0c953ce.tar.gz"
    TEXT_ENCODER_URL="https://rtcos-1394285684.cos.ap-nanjing.myqcloud.com/models/wan-text-encoder.tar.gz"
    # 注：COS 上使用固定名称，不含版本号（简化管理）
    SOURCE_NAME="COS (中国)"
else
    # Google Cloud Storage（海外源）
    BASE_URL="https://storage.googleapis.com/lxcpublic/krea-models-base-6b5d204f.tar.gz"
    FP8_URL="https://storage.googleapis.com/lxcpublic/krea-models-fp8-f0c953ce.tar.gz"
    TEXT_ENCODER_URL="https://storage.googleapis.com/lxcpublic/wan-text-encoder.tar.gz"
    SOURCE_NAME="GCS"
fi

echo "📡 下载源: $SOURCE_NAME"
echo ""

TARGET_DIR="./tmp/.hf_home/hub"

# 获取本地模型版本
get_local_version() {
    local model_dir=$1
    local refs_file="$model_dir/refs/main"
    
    if [ -f "$refs_file" ]; then
        cat "$refs_file" | head -c 8
    else
        local snapshot_dir=$(ls -1 "$model_dir/snapshots" 2>/dev/null | head -1)
        if [ -n "$snapshot_dir" ]; then
            echo "$snapshot_dir" | head -c 8
        else
            echo ""
        fi
    fi
}

# 从 URL 提取版本号
get_url_version() {
    local url=$1
    # 从 krea-models-base-6b5d204f.tar.gz 提取 6b5d204f
    echo "$url" | grep -oP '(?<=-)[a-f0-9]{8}(?=\.tar\.gz)' || echo ""
}

# 下载函数
download_model() {
    local url=$1
    local name=$2
    local model_dir=$3
    local temp_file="./tmp/krea-model-temp.tar.gz"
    
    # 检查版本是否匹配
    local url_version=$(get_url_version "$url")
    local local_version=$(get_local_version "$model_dir")
    
    if [ -n "$local_version" ] && [ "$local_version" = "$url_version" ]; then
        echo "✅ $name 已是最新版本 ($local_version)，跳过"
        return 0
    elif [ -n "$local_version" ]; then
        echo "📥 $name 版本更新: $local_version -> $url_version"
    fi
    
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
BASE_MODEL_DIR="$TARGET_DIR/models--krea--krea-realtime-video"
if [ -n "$BASE_URL" ]; then
    download_model "$BASE_URL" "基础模型" "$BASE_MODEL_DIR" || echo "   将在运行时从 HuggingFace 下载"
else
    if [ ! -d "$BASE_MODEL_DIR" ]; then
        echo "⚠️  基础模型 URL 未配置，将从 HuggingFace 下载"
    else
        echo "✅ 基础模型已存在"
    fi
fi

echo ""

# 下载 FP8 模型（如果指定 --fp8）
if [ "$DOWNLOAD_FP8" = true ]; then
    FP8_MODEL_DIR="$TARGET_DIR/models--6chan--krea-realtime-video-fp8"
    if [ -n "$FP8_URL" ]; then
        download_model "$FP8_URL" "FP8 模型" "$FP8_MODEL_DIR" || echo "   将在运行时从 HuggingFace 下载"
    else
        if [ ! -d "$FP8_MODEL_DIR" ]; then
            echo "⚠️  FP8 模型 URL 未配置，将从 HuggingFace 下载"
        else
            echo "✅ FP8 模型已存在"
        fi
    fi
fi

# 下载 Text Encoder（Wan-AI）- 注意：存放在 transformers 目录
TEXT_ENCODER_TRANSFORMERS_DIR="./tmp/.hf_home/transformers/models--Wan-AI--Wan2.1-T2V-14B-Diffusers"
TEXT_ENCODER_HUB_DIR="$TARGET_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers"

# 检查 text_encoder 是否完整（至少需要 10GB）
TEXT_ENCODER_SIZE_T=$(du -sm "$TEXT_ENCODER_TRANSFORMERS_DIR" 2>/dev/null | cut -f1 || echo "0")
TEXT_ENCODER_SIZE_H=$(du -sm "$TEXT_ENCODER_HUB_DIR" 2>/dev/null | cut -f1 || echo "0")

if [ "$TEXT_ENCODER_SIZE_T" -lt 10000 ] && [ "$TEXT_ENCODER_SIZE_H" -lt 10000 ]; then
    if [ -n "$TEXT_ENCODER_URL" ]; then
        echo ""
        echo "📥 下载 Text Encoder (Wan-AI)..."
        echo "   URL: $TEXT_ENCODER_URL"
        
        temp_file="./tmp/wan-text-encoder-temp.tar.gz"
        success=false
        
        if command -v wget &> /dev/null; then
            wget -O "$temp_file" "$TEXT_ENCODER_URL" && success=true
        elif command -v curl &> /dev/null; then
            curl -L -o "$temp_file" "$TEXT_ENCODER_URL" && success=true
        fi
        
        if [ "$success" = true ]; then
            echo "   ✅ 下载成功"
            echo "   📦 解压到 transformers 缓存..."
            mkdir -p ./tmp/.hf_home/transformers
            tar -xzf "$temp_file" -C ./tmp/.hf_home/transformers
            rm -f "$temp_file"
            echo "   ✅ 完成"
        else
            echo "   ⚠️  下载失败，将在运行时从 HuggingFace 下载"
        fi
    else
        echo "⚠️  Text Encoder 不完整，将在运行时从 HuggingFace 下载"
    fi
else
    if [ "$TEXT_ENCODER_SIZE_T" -gt 10000 ]; then
        echo "✅ Text Encoder 已存在 (transformers: ${TEXT_ENCODER_SIZE_T}MB)"
    else
        echo "✅ Text Encoder 已存在 (hub: ${TEXT_ENCODER_SIZE_H}MB)"
    fi
fi

echo ""
echo "✅ 模型检查完成"
