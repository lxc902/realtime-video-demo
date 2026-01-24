"""
量化模块 - 支持多种精度/量化方式加载 KREA 模型

支持的精度:
- bf16: BF16 标准加载 (~54GB 显存)
- fp8:  FP8 量化 (~24GB 显存)
- int8: INT8 量化 (~28GB 显存)
- int4: INT4 量化 (~16GB 显存)
"""
from .bf16 import load_bf16
from .fp8 import load_fp8
from .int8 import load_int8
from .int4 import load_int4


def load_model_with_quantization(pipe, repo_id, device, dtype, quantization=None):
    """根据量化类型加载模型
    
    Args:
        pipe: ModularPipeline 实例
        repo_id: 模型仓库 ID
        device: 目标设备
        dtype: 基础数据类型
        quantization: 量化类型 (None/"bf16", "fp8", "int8", "int4")
    
    Returns:
        加载完成的 pipe
    """
    if quantization == "fp8":
        return load_fp8(pipe, repo_id, device, dtype)
    elif quantization == "int8":
        return load_int8(pipe, repo_id, device, dtype)
    elif quantization == "int4":
        return load_int4(pipe, repo_id, device, dtype)
    else:
        return load_bf16(pipe, repo_id, device, dtype)
