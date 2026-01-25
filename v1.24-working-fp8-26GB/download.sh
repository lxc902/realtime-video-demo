#!/bin/bash

set -e

echo "==========================================="
echo "下载 KREA 模型"
echo "==========================================="
echo ""

# 配置 - GCS 备份 URL（使用版本号命名）
# 格式: krea-models-base-{commit_hash前8位}.tar.gz
GCS_BASE_URL="https://storage.googleapis.com/lxcpublic/krea-models-base-6b5d204f.tar.gz"
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
if [ -n "$GCS_BASE_URL" ]; then
    download_model "$GCS_BASE_URL" "基础模型" "$BASE_MODEL_DIR" || echo "   将在运行时从 HuggingFace 下载"
else
    if [ ! -d "$BASE_MODEL_DIR" ]; then
        echo "⚠️  基础模型 GCS URL 未配置，将从 HuggingFace 下载"
    else
        echo "✅ 基础模型已存在"
    fi
fi

echo ""

# 下载 FP8 模型（如果指定 --fp8）
if [ "$DOWNLOAD_FP8" = true ]; then
    FP8_MODEL_DIR="$TARGET_DIR/models--6chan--krea-realtime-video-fp8"
    if [ -n "$GCS_FP8_URL" ]; then
        download_model "$GCS_FP8_URL" "FP8 模型" "$FP8_MODEL_DIR" || echo "   将在运行时从 HuggingFace 下载"
    else
        if [ ! -d "$FP8_MODEL_DIR" ]; then
            echo "⚠️  FP8 模型 GCS URL 未配置，将从 HuggingFace 下载"
        else
            echo "✅ FP8 模型已存在"
        fi
    fi
fi

echo ""
echo "✅ 模型检查完成"
