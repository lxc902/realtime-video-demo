"""
配置文件 - 模型路径和量化设置
"""
import os

# 模型路径配置
# 可以设置为：
# 1. None - 从 HuggingFace 自动下载 (默认)
# 2. 本地路径 - 例如 "/models/krea-realtime-video"
# 3. 环境变量 - 从 MODEL_PATH 环境变量读取

MODEL_PATH = os.getenv("MODEL_PATH", None)

# 如果设置了本地路径，检查是否存在
if MODEL_PATH and not os.path.exists(MODEL_PATH):
    print(f"警告: 指定的模型路径不存在: {MODEL_PATH}")
    print(f"将使用 HuggingFace 默认路径")
    MODEL_PATH = None

# 量化配置
# 可选值: None, "int8", "int4"
# - None: 不量化 (需要 ~54GB+ 显存)
# - "int8": 8位量化 (需要 ~24GB 显存)
# - "int4": 4位量化 (需要 ~12GB 显存)
QUANTIZATION = os.getenv("QUANTIZATION", None)

if QUANTIZATION:
    QUANTIZATION = QUANTIZATION.lower()
    if QUANTIZATION not in ("int8", "int4"):
        print(f"警告: 不支持的量化类型: {QUANTIZATION}")
        print(f"支持的类型: int8, int4")
        QUANTIZATION = None
