"""
FP8 优化模块 - 基于 ComfyUI-WanVideoWrapper 的实现
只对 Linear 层应用 FP8 计算，保持 Conv/Norm 等层为原精度
"""
import torch
import torch.nn as nn


def fp8_linear_forward(cls, base_dtype, input):
    """FP8 Linear 层的 forward 函数
    
    基于 ComfyUI 和 MinusZoneAI 的 fp8_linear 优化
    使用 torch._scaled_mm 进行 FP8 矩阵乘法
    """
    weight_dtype = cls.weight.dtype
    
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            input_shape = input.shape

            # 获取或创建 scale_weight
            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device).squeeze()

            scale_input = torch.ones((), device=input.device, dtype=torch.float32)

            # Clamp 输入到 FP8 e4m3fn 的有效范围
            input = torch.clamp(input, min=-448, max=448, out=input)
            
            # 转换输入为 FP8（始终使用 e4m3fn，因为 e5m2 * e5m2 不支持）
            inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()

            bias = cls.bias.to(base_dtype) if cls.bias is not None else None

            # 使用 scaled_mm 进行 FP8 矩阵乘法
            o = torch._scaled_mm(
                inn, 
                cls.weight.t(), 
                out_dtype=base_dtype, 
                bias=bias, 
                scale_a=scale_input, 
                scale_b=scale_weight
            )

            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            # 非 3D 输入，回退到原始 forward
            return cls.original_forward(input.to(base_dtype))
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, base_dtype, params_to_keep=None, scale_weight_keys=None):
    """将模型中的 Linear 层转换为 FP8 优化版本
    
    Args:
        module: 要转换的模型
        base_dtype: 基础数据类型（输出类型）
        params_to_keep: 需要保持原精度的参数名关键字集合
        scale_weight_keys: scale_weight 字典
    """
    if params_to_keep is None:
        # 这些层保持原精度，不做 FP8 量化
        params_to_keep = {
            "norm", "bias", "time_in", "patch_embedding", "time_", 
            "img_emb", "modulation", "text_embedding", "adapter", 
            "add", "ref_conv", "audio_proj"
        }
    
    print("🔧 启用 FP8 矩阵乘法优化...")
    converted_count = 0
    skipped_count = 0
    
    for name, submodule in module.named_modules():
        # 跳过需要保持原精度的层
        if any(keyword in name for keyword in params_to_keep):
            skipped_count += 1
            continue
            
        if isinstance(submodule, nn.Linear):
            # 检查是否有 scale_weight
            if scale_weight_keys is not None:
                scale_key = f"{name}.scale_weight"
                if scale_key in scale_weight_keys:
                    setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())
            
            # 保存原始 forward 并替换为 FP8 版本
            original_forward = submodule.forward
            setattr(submodule, "original_forward", original_forward)
            setattr(submodule, "forward", 
                    lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))
            converted_count += 1
    
    print(f"   ✅ 已转换 {converted_count} 个 Linear 层为 FP8")
    print(f"   ⏭️  跳过 {skipped_count} 个特殊层（保持原精度）")


def load_fp8_weights(model, state_dict, base_dtype=torch.bfloat16, device="cuda"):
    """加载 FP8 权重到模型
    
    Args:
        model: 模型实例
        state_dict: FP8 权重字典
        base_dtype: 非 FP8 层的数据类型
        device: 目标设备
    
    Returns:
        scale_weights: scale_weight 字典
    """
    # 需要保持原精度的参数
    params_to_keep = {
        "norm", "bias", "time_in", "patch_embedding", "time_", 
        "img_emb", "modulation", "text_embedding", "adapter", 
        "add", "ref_conv", "audio_proj"
    }
    
    # 提取 scale_weights
    scale_weights = {}
    for k, v in state_dict.items():
        if k.endswith(".scale_weight") or k.endswith(".weight_scale"):
            scale_weights[k.replace(".weight_scale", ".scale_weight")] = v.to(device, torch.float32)
    
    print(f"   找到 {len(scale_weights)} 个 scale_weight")
    
    # 加载权重
    model_state = model.state_dict()
    loaded_count = 0
    
    for name, param in state_dict.items():
        if name in model_state:
            # 判断是否需要保持原精度
            keep_original = any(keyword in name for keyword in params_to_keep)
            
            if keep_original or not isinstance(param, torch.Tensor):
                # 转换为 base_dtype
                if isinstance(param, torch.Tensor):
                    param = param.to(device, base_dtype)
            else:
                # FP8 权重保持原样
                param = param.to(device)
            
            try:
                model_state[name].copy_(param)
                loaded_count += 1
            except Exception as e:
                print(f"   ⚠️  无法加载 {name}: {e}")
    
    print(f"   ✅ 已加载 {loaded_count} 个参数")
    
    return scale_weights


def check_fp8_support():
    """检查当前硬件是否支持 FP8 计算
    
    Returns:
        tuple: (supports_fp8, compute_capability, message)
    """
    if not torch.cuda.is_available():
        return False, None, "CUDA 不可用"
    
    major, minor = torch.cuda.get_device_capability()
    compute_cap = f"{major}.{minor}"
    
    # FP8 matmul 需要 CUDA compute capability >= 8.9 (RTX 4000 系列及以上)
    if (major, minor) >= (8, 9):
        return True, compute_cap, f"完全支持 FP8 (Compute Capability {compute_cap})"
    elif (major, minor) >= (8, 0):
        return True, compute_cap, f"部分支持 FP8 (Compute Capability {compute_cap})，推荐 RTX 4000+ 系列"
    else:
        return False, compute_cap, f"不支持 FP8 (Compute Capability {compute_cap})，需要 >= 8.0"
