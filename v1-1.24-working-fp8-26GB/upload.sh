#!/bin/bash

set -e

echo "==========================================="
echo "KREA 模型备份到 Google Cloud Storage"
echo "==========================================="
echo ""

# 配置
BUCKET="gs://lxcpublic"
MODEL_DIR="./tmp/.hf_home/hub"

# 解析参数
FORCE_UPLOAD=false
for arg in "$@"; do
    case $arg in
        --force)
            FORCE_UPLOAD=true
            ;;
    esac
done

# 获取模型版本（从 HuggingFace refs/main 读取 commit hash）
get_model_version() {
    local model_dir=$1
    local refs_file="$model_dir/refs/main"
    
    if [ -f "$refs_file" ]; then
        # 读取 commit hash，取前 8 位
        cat "$refs_file" | head -c 8
    else
        # 如果没有 refs 文件，使用 snapshots 目录名
        local snapshot_dir=$(ls -1 "$model_dir/snapshots" 2>/dev/null | head -1)
        if [ -n "$snapshot_dir" ]; then
            echo "$snapshot_dir" | head -c 8
        else
            echo "unknown"
        fi
    fi
}

# 检查 gsutil
install_gsutil() {
    if ! command -v gsutil &> /dev/null; then
        echo "⚠️  gsutil 未安装，正在自动安装..."
        if command -v pip &> /dev/null; then
            pip install gsutil
        elif command -v pip3 &> /dev/null; then
            pip3 install gsutil
        else
            apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl
            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
            echo "deb https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list
            apt-get update && apt-get install -y google-cloud-sdk
        fi
        echo "✅ gsutil 安装完成"
        echo ""
    fi
}

# 检查 GCS 上是否已存在
check_gcs_exists() {
    local backup_name=$1
    gsutil -q stat "$BUCKET/$backup_name" 2>/dev/null
    return $?
}

# 上传单个模型包
upload_model() {
    local name=$1
    local backup_name=$2
    shift 2
    local dirs=("$@")
    
    # 检查是否已存在
    if [ "$FORCE_UPLOAD" = false ] && check_gcs_exists "$backup_name"; then
        echo "✅ $name 已存在于 GCS，跳过上传"
        echo "   📥 https://storage.googleapis.com/lxcpublic/$backup_name"
        echo "   (使用 --force 强制重新上传)"
        echo ""
        return 0
    fi
    
    echo "📦 打包 $name..."
    echo "   目标: $BUCKET/$backup_name"
    
    cd $MODEL_DIR
    tar -czf - "${dirs[@]}" \
        | gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
                 cp - $BUCKET/$backup_name
    cd - > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 上传成功"
        echo "   📥 https://storage.googleapis.com/lxcpublic/$backup_name"
        echo ""
        return 0
    else
        echo "   ❌ 上传失败"
        return 1
    fi
}

# 检查有哪些模型
HAS_BASE=false
HAS_FP8=false
HAS_TEXT_ENCODER=false
BASE_VERSION=""
FP8_VERSION=""
TEXT_ENCODER_VERSION=""

if [ -d "$MODEL_DIR/models--krea--krea-realtime-video" ]; then
    HAS_BASE=true
    BASE_VERSION=$(get_model_version "$MODEL_DIR/models--krea--krea-realtime-video")
fi

if [ -d "$MODEL_DIR/models--6chan--krea-realtime-video-fp8" ]; then
    HAS_FP8=true
    FP8_VERSION=$(get_model_version "$MODEL_DIR/models--6chan--krea-realtime-video-fp8")
fi

# Text Encoder 可能在 hub/ 或 transformers/ 目录
TEXT_ENCODER_HUB_DIR="$MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers"
TEXT_ENCODER_TRANSFORMERS_DIR="./tmp/.hf_home/transformers/models--Wan-AI--Wan2.1-T2V-14B-Diffusers"

if [ -d "$TEXT_ENCODER_TRANSFORMERS_DIR" ]; then
    # transformers 库缓存目录（大文件在这里）
    TEXT_ENCODER_SIZE=$(du -sm "$TEXT_ENCODER_TRANSFORMERS_DIR" 2>/dev/null | cut -f1 || echo "0")
    if [ "$TEXT_ENCODER_SIZE" -gt 10000 ]; then
        HAS_TEXT_ENCODER=true
        TEXT_ENCODER_VERSION=$(get_model_version "$TEXT_ENCODER_TRANSFORMERS_DIR")
        TEXT_ENCODER_LOCATION="transformers"
    else
        echo "⚠️  Text Encoder (transformers) 不完整 (${TEXT_ENCODER_SIZE}MB < 10GB)"
    fi
fi

if [ "$HAS_TEXT_ENCODER" = false ] && [ -d "$TEXT_ENCODER_HUB_DIR" ]; then
    # hub 目录备选
    TEXT_ENCODER_SIZE=$(du -sm "$TEXT_ENCODER_HUB_DIR" 2>/dev/null | cut -f1 || echo "0")
    if [ "$TEXT_ENCODER_SIZE" -gt 10000 ]; then
        HAS_TEXT_ENCODER=true
        TEXT_ENCODER_VERSION=$(get_model_version "$TEXT_ENCODER_HUB_DIR")
        TEXT_ENCODER_LOCATION="hub"
    else
        echo "⚠️  Text Encoder (hub) 不完整 (${TEXT_ENCODER_SIZE}MB < 10GB)，跳过"
    fi
fi

if [ "$HAS_BASE" = false ] && [ "$HAS_FP8" = false ] && [ "$HAS_TEXT_ENCODER" = false ]; then
    echo "❌ 错误: 没有找到任何模型"
    echo "   请先运行 bash run.sh 下载模型"
    exit 1
fi

# 显示要备份的模型
echo "📁 检测到以下模型:"
if [ "$HAS_BASE" = true ]; then
    echo "   [BASE] 基础模型 (版本: $BASE_VERSION):"
    du -sh $MODEL_DIR/models--krea--krea-realtime-video 2>/dev/null || true
fi
if [ "$HAS_FP8" = true ]; then
    echo "   [FP8] FP8 量化模型 (版本: $FP8_VERSION):"
    du -sh $MODEL_DIR/models--6chan--krea-realtime-video-fp8 2>/dev/null || true
fi
if [ "$HAS_TEXT_ENCODER" = true ]; then
    echo "   [TEXT] Text Encoder (版本: $TEXT_ENCODER_VERSION):"
    du -sh $MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers 2>/dev/null || true
fi
echo ""

if [ "$FORCE_UPLOAD" = true ]; then
    echo "⚠️  强制上传模式：将覆盖已存在的文件"
    echo ""
fi

# 安装 gsutil
install_gsutil

# 上传基础模型（不再包含 text encoder，因为太大）
if [ "$HAS_BASE" = true ]; then
    BASE_BACKUP="krea-models-base-${BASE_VERSION}.tar.gz"
    upload_model "基础模型" "$BASE_BACKUP" "models--krea--krea-realtime-video"
fi

# 上传 FP8 模型
if [ "$HAS_FP8" = true ]; then
    FP8_BACKUP="krea-models-fp8-${FP8_VERSION}.tar.gz"
    upload_model "FP8 模型" "$FP8_BACKUP" "models--6chan--krea-realtime-video-fp8"
fi

# 上传 Wan-AI 模型（Text Encoder + VAE，约 20GB）
# Text Encoder 在 transformers/ 目录，VAE 在 hub/ 目录，需要都打包
if [ "$HAS_TEXT_ENCODER" = true ]; then
    WAN_AI_BACKUP="wan-ai-models-${TEXT_ENCODER_VERSION}.tar.gz"
    
    echo "📦 打包 Wan-AI 模型 (Text Encoder + VAE)..."
    echo "   目标: $BUCKET/$WAN_AI_BACKUP"
    
    # 检查是否已存在
    if [ "$FORCE_UPLOAD" = false ] && check_gcs_exists "$WAN_AI_BACKUP"; then
        echo "✅ Wan-AI 模型已存在于 GCS，跳过上传"
        echo "   📥 https://storage.googleapis.com/lxcpublic/$WAN_AI_BACKUP"
    else
        # 创建临时目录结构并打包
        TEMP_PACK_DIR="./tmp/wan_ai_pack"
        rm -rf "$TEMP_PACK_DIR"
        mkdir -p "$TEMP_PACK_DIR/hub" "$TEMP_PACK_DIR/transformers"
        
        # 复制 hub 目录（VAE）
        if [ -d "$MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers" ]; then
            cp -r "$MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers" "$TEMP_PACK_DIR/hub/"
            echo "   ✓ 包含 hub/VAE"
        fi
        
        # 复制 transformers 目录（Text Encoder）
        if [ -d "./tmp/.hf_home/transformers/models--Wan-AI--Wan2.1-T2V-14B-Diffusers" ]; then
            cp -r "./tmp/.hf_home/transformers/models--Wan-AI--Wan2.1-T2V-14B-Diffusers" "$TEMP_PACK_DIR/transformers/"
            echo "   ✓ 包含 transformers/Text Encoder"
        fi
        
        # 打包并上传
        cd "$TEMP_PACK_DIR"
        tar -czf - hub transformers \
            | gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
                     cp - $BUCKET/$WAN_AI_BACKUP
        cd - > /dev/null
        
        # 清理临时目录
        rm -rf "$TEMP_PACK_DIR"
        
        echo "   ✅ 上传成功"
        echo "   📥 https://storage.googleapis.com/lxcpublic/$WAN_AI_BACKUP"
    fi
fi

echo "==========================================="
echo "✅ 全部完成！"
echo ""
echo "📝 download.sh 中的 URL:"
if [ "$HAS_BASE" = true ]; then
    echo "   BASE_URL: https://storage.googleapis.com/lxcpublic/$BASE_BACKUP"
fi
if [ "$HAS_FP8" = true ]; then
    echo "   FP8_URL: https://storage.googleapis.com/lxcpublic/$FP8_BACKUP"
fi
if [ "$HAS_TEXT_ENCODER" = true ]; then
    echo "   WAN_AI_URL: https://storage.googleapis.com/lxcpublic/$WAN_AI_BACKUP"
fi
echo ""
echo "⚠️  上传完成后，请运行 move_gcs_to_cos.sh 迁移到 COS"
echo "==========================================="
