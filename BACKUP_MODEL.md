# 模型备份和使用指南

## 步骤 1: 备份已下载的模型

### 1.1 找到模型位置
```bash
# 模型通常在这里
ls -lh ~/.cache/huggingface/hub/models--krea--krea-realtime-video/
```

### 1.2 打包模型
```bash
cd ~/.cache/huggingface/hub
tar -czf krea-realtime-video.tar.gz models--krea--krea-realtime-video/
```

### 1.3 上传到对象存储
```bash
# AWS S3
aws s3 cp krea-realtime-video.tar.gz s3://your-bucket/models/

# 阿里云 OSS
ossutil cp krea-realtime-video.tar.gz oss://your-bucket/models/

# MinIO
mc cp krea-realtime-video.tar.gz myminio/your-bucket/models/

# Rclone (通用)
rclone copy krea-realtime-video.tar.gz your-remote:your-bucket/models/
```

## 步骤 2: 使用备份的模型

### 方法 1: 从对象存储下载到本地

```bash
# 创建模型目录
mkdir -p /data/models

# 下载并解压
cd /data/models
aws s3 cp s3://your-bucket/models/krea-realtime-video.tar.gz .
tar -xzf krea-realtime-video.tar.gz

# 解压后的路径应该是
# /data/models/models--krea--krea-realtime-video/
```

### 方法 2: 设置环境变量

```bash
# 方式 A: 启动时指定
export MODEL_PATH="/data/models/models--krea--krea-realtime-video"
bash run.sh

# 方式 B: 修改 config.py
# 编辑 config.py，设置:
# MODEL_PATH = "/data/models/models--krea--krea-realtime-video"
```

### 方法 3: 使用对象存储作为缓存

如果你的对象存储支持 S3 协议，可以挂载为本地文件系统：

```bash
# 使用 s3fs 挂载
apt install s3fs
mkdir -p /mnt/s3models
s3fs your-bucket /mnt/s3models -o url=https://your-endpoint.com

# 设置模型路径
export MODEL_PATH="/mnt/s3models/models/models--krea--krea-realtime-video"
bash run.sh
```

## 步骤 3: 验证

```bash
# 启动服务后检查
curl http://localhost:7860/health

# 应该看到:
# {
#   "status": "ok",
#   "mode": "local_gpu",
#   "model_loaded": true,
#   "ready": true
# }
```

## 目录结构说明

备份的模型目录结构：
```
models--krea--krea-realtime-video/
├── snapshots/
│   └── <commit-hash>/
│       ├── config.json
│       ├── modular_config.json
│       ├── modular_blocks.py
│       ├── vae/
│       │   └── diffusion_pytorch_model.safetensors
│       └── ... (其他文件)
└── refs/
```

## 注意事项

1. **完整性检查**: 解压后确保所有文件都存在
2. **权限设置**: 确保文件可读 `chmod -R 755 /path/to/model`
3. **磁盘空间**: 模型约 14GB，确保有足够空间
4. **版本管理**: 建议在文件名中包含日期或版本号

## 故障排除

### 模型加载失败
```bash
# 检查路径是否正确
ls -lh $MODEL_PATH

# 检查权限
ls -ld $MODEL_PATH
```

### 文件缺失
如果某些文件缺失，可能需要重新下载：
```bash
# 临时使用 HuggingFace
unset MODEL_PATH
bash run.sh
```
