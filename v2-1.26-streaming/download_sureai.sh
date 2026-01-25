#!/bin/bash
set -e

PEM="$HOME/.ssh/zhaoxianghui_heyuan.pem"
REMOTE_HOST="root@47.115.222.118"

LOCAL_BASE="$HOME/Downloads/sureai"
TMP_REMOTE_DIR="/tmp/kode-backend-runtime"
CONTAINER_NAME="kode-backend"

# 修正权限
chmod 600 "$PEM"

# 创建本地目录
mkdir -p "$LOCAL_BASE"

echo "==> 1. 下载前端代码"
rsync -avz --progress \
  -e "ssh -i $PEM" \
  $REMOTE_HOST:/var/www/sureai-web \
  "$LOCAL_BASE/"

echo "==> 2. 下载 nginx 配置"
rsync -avz --progress \
  -e "ssh -i $PEM" \
  $REMOTE_HOST:/etc/nginx/conf.d \
  "$LOCAL_BASE/nginx-conf.d"

echo "==> 3. 从 Docker 容器导出后端运行时代码 (/app)"
ssh -i "$PEM" $REMOTE_HOST << 'EOF'
set -e
rm -rf /tmp/kode-backend-runtime
mkdir -p /tmp/kode-backend-runtime
docker cp kode-backend:/app /tmp/kode-backend-runtime/app
EOF

echo "==> 4. 拉取 Docker 中的后端代码到本地（跳过 uploads / node_modules）"
rsync -av --progress \
  --exclude 'uploads/' \
  --exclude 'node_modules/' \
  --exclude '.kode/' \
  -e "ssh -i $PEM -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
  $REMOTE_HOST:$TMP_REMOTE_DIR \
  "$LOCAL_BASE/backend-runtime"


echo "✅ 全部完成"
echo "前端:        $LOCAL_BASE/sureai-web"
echo "Nginx:       $LOCAL_BASE/nginx-conf.d"
echo "后端(容器):  $LOCAL_BASE/backend-runtime/app"
