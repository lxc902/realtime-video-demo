# KREA Realtime Video - 本地 GPU 版本

本项目已修改为使用本地 GPU 运行 KREA 模型，无需 FAL API Key。

## 系统要求

- **GPU**: NVIDIA RTX 6000 Ada 或同等性能 GPU
- **VRAM**: 至少 24GB (推荐)
- **CUDA**: 12.0 或更高
- **系统**: Linux (Ubuntu 推荐)
- **Python**: 3.10+

## 快速开始

### 一键启动

```bash
bash run.sh
```

就这么简单！`run.sh` 会自动：
1. ✅ 检测并安装所有依赖（首次运行）
2. ✅ 显示 GPU 信息
3. ✅ 下载 KREA 模型（首次运行，~14GB）
4. ✅ 启动服务器

**首次运行需要**：
- 良好的网络连接
- 约 10-15 分钟（依赖安装 + 模型下载）
- 足够的磁盘空间 (~20GB)

**后续启动**：
- 只需 1-2 分钟（加载模型到 GPU）

### 访问界面

打开浏览器访问: http://YOUR_SERVER_IP:7860

## 性能

- **RTX 6000 Ada**: ~8-10 fps (预估)
- **A100**: ~10-12 fps
- **B200**: ~11 fps (官方测试)

## 文件说明

- `run.sh` - 🚀 一键启动脚本（自动安装依赖）
- `upload.sh` - ☁️ 备份模型到 Google Cloud Storage
- `download.sh` - 📥 从 Google Cloud Storage 恢复模型
- `app_local.py` - 使用本地 GPU 的 FastAPI 服务器
- `local_inference.py` - KREA 模型推理模块
- `config.py` - 配置文件（支持自定义模型路径）
- `app.py` - 原 FAL API 版本 (备份，已废弃)

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

## 停止服务

在运行终端按 `Ctrl+C` 即可停止服务。

## 许可证

KREA 模型遵循其原始许可证。详见: https://huggingface.co/krea/krea-realtime-video
