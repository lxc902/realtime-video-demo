# KREA Realtime Video - 本地 GPU 版本

本项目已修改为使用本地 GPU 运行 KREA 模型，无需 FAL API Key。

## 系统要求

- **GPU**: NVIDIA RTX 6000 Ada 或同等性能 GPU
- **VRAM**: 至少 24GB (推荐)
- **CUDA**: 12.0 或更高
- **系统**: Linux (Ubuntu 推荐)
- **Python**: 3.10+

## 快速开始

### 1. 安装依赖

```bash
bash install_local.sh
```

这将安装：
- PyTorch (CUDA 版本)
- Diffusers (最新开发版)
- 其他必要依赖

### 2. 启动服务

```bash
bash start_local.sh
```

**注意**: 首次运行会自动从 HuggingFace 下载模型 (~14GB)，需要：
- 良好的网络连接
- 约 5-10 分钟下载时间
- 足够的磁盘空间 (~20GB)

### 3. 访问界面

打开浏览器访问: http://YOUR_SERVER_IP:7860

## 性能

- **RTX 6000 Ada**: ~8-10 fps (预估)
- **A100**: ~10-12 fps
- **B200**: ~11 fps (官方测试)

## 文件说明

- `app_local.py` - 使用本地 GPU 的 FastAPI 服务器
- `local_inference.py` - KREA 模型推理模块
- `install_local.sh` - 依赖安装脚本
- `start_local.sh` - 服务启动脚本
- `app.py` - 原 FAL API 版本 (备份)

## 故障排除

### CUDA Out of Memory

如果显存不足：
```python
# 在 local_inference.py 中修改:
dtype=torch.float16  # 使用 float16 而不是 bfloat16
```

### 模型下载失败

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 生成速度慢

1. 确保安装了 flash-attention
2. 检查是否启用了 `torch.compile`
3. 减少 `num_inference_steps` (默认 4，可以降到 2)

## 切换回 FAL API 模式

如果需要切换回 FAL API:
```bash
# 使用原来的启动脚本
bash start.sh
```

## 许可证

KREA 模型遵循其原始许可证。详见: https://huggingface.co/krea/krea-realtime-video
