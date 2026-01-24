#!/bin/bash

set -e

echo "==========================================="
echo "KREA 模型备份到 Google Cloud Storage"
echo "==========================================="
echo ""

# 配置
BUCKET="gs://lxcpublic"
MODEL_DIR="./tmp/.hf_home/hub"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

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

# 上传单个模型包
upload_model() {
    local name=$1
    local backup_name=$2
    shift 2
    local dirs=("$@")
    
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

if [ -d "$MODEL_DIR/models--krea--krea-realtime-video" ]; then
    HAS_BASE=true
fi

if [ -d "$MODEL_DIR/models--6chan--krea-realtime-video-fp8" ]; then
    HAS_FP8=true
fi

if [ "$HAS_BASE" = false ] && [ "$HAS_FP8" = false ]; then
    echo "❌ 错误: 没有找到任何模型"
    echo "   请先运行 bash run.sh 下载模型"
    exit 1
fi

# 显示要备份的模型
echo "📁 检测到以下模型:"
if [ "$HAS_BASE" = true ]; then
    echo "   [BASE] 基础模型:"
    du -sh $MODEL_DIR/models--krea--krea-realtime-video 2>/dev/null || true
    du -sh $MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers 2>/dev/null || true
fi
if [ "$HAS_FP8" = true ]; then
    echo "   [FP8] FP8 量化模型:"
    du -sh $MODEL_DIR/models--6chan--krea-realtime-video-fp8 2>/dev/null || true
fi
echo ""

# 安装 gsutil
install_gsutil

# 上传基础模型
if [ "$HAS_BASE" = true ]; then
    BASE_BACKUP="krea-models-base-$TIMESTAMP.tar.gz"
    BASE_DIRS=("models--krea--krea-realtime-video")
    
    # 如果有 text encoder 也一起打包
    if [ -d "$MODEL_DIR/models--Wan-AI--Wan2.1-T2V-14B-Diffusers" ]; then
        BASE_DIRS+=("models--Wan-AI--Wan2.1-T2V-14B-Diffusers")
    fi
    
    upload_model "基础模型" "$BASE_BACKUP" "${BASE_DIRS[@]}"
fi

# 上传 FP8 模型
if [ "$HAS_FP8" = true ]; then
    FP8_BACKUP="krea-models-fp8-$TIMESTAMP.tar.gz"
    upload_model "FP8 模型" "$FP8_BACKUP" "models--6chan--krea-realtime-video-fp8"
fi

echo "==========================================="
echo "✅ 全部完成！"
echo ""
echo "📝 请更新 download.sh 中的 URL:"
if [ "$HAS_BASE" = true ]; then
    echo "   GCS_BASE_URL=\"https://storage.googleapis.com/lxcpublic/$BASE_BACKUP\""
fi
if [ "$HAS_FP8" = true ]; then
    echo "   GCS_FP8_URL=\"https://storage.googleapis.com/lxcpublic/$FP8_BACKUP\""
fi
echo "==========================================="
