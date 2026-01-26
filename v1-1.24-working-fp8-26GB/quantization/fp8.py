"""
FP8 量化加载
基于 ComfyUI-WanVideoWrapper 的实现
需要 ~24GB 显存，需要 Compute Capability >= 8.0
"""
import torch
import torch.nn as nn


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


def fp8_linear_forward(cls, base_dtype, input):
    """FP8 Linear 层的 forward 函数
    
    使用 torch._scaled_mm 进行 FP8 矩阵乘法
    """
    weight_dtype = cls.weight.dtype
    
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # 获取或创建 scale_weight
        scale_weight = getattr(cls, 'scale_weight', None)
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device).squeeze()

        # 复用预创建的 scale_input，避免每次 forward 创建新张量
        scale_input = getattr(cls, '_scale_input_cache', None)
        if scale_input is None or scale_input.device != input.device:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            cls._scale_input_cache = scale_input
        
        # Clamp 输入到 FP8 e4m3fn 的有效范围
        input = torch.clamp(input, min=-448, max=448)
        
        bias = cls.bias.to(base_dtype) if cls.bias is not None else None
        
        if len(input.shape) == 3:
            # 3D 输入: [batch, seq, features]
            input_shape = input.shape
            inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
            
            o = torch._scaled_mm(
                inn, 
                cls.weight.t(), 
                out_dtype=base_dtype, 
                bias=bias, 
                scale_a=scale_input, 
                scale_b=scale_weight
            )
            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        
        elif len(input.shape) == 2:
            # 2D 输入: [batch, features]
            inn = input.to(torch.float8_e4m3fn).contiguous()
            
            o = torch._scaled_mm(
                inn, 
                cls.weight.t(), 
                out_dtype=base_dtype, 
                bias=bias, 
                scale_a=scale_input, 
                scale_b=scale_weight
            )
            return o
        
        else:
            # 其他维度：将权重转换为 base_dtype 进行计算
            weight_bf16 = cls.weight.to(base_dtype)
            return torch.nn.functional.linear(input.to(base_dtype), weight_bf16, bias)
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
        params_to_keep = {
            "norm", "bias", "time_in", "patch_embedding", "time_", 
            "img_emb", "modulation", "text_embedding", "adapter", 
            "add", "ref_conv", "audio_proj"
        }
    
    print("🔧 启用 FP8 矩阵乘法优化...")
    converted_count = 0
    skipped_count = 0
    
    for name, submodule in module.named_modules():
        if any(keyword in name for keyword in params_to_keep):
            skipped_count += 1
            continue
            
        if isinstance(submodule, nn.Linear):
            if scale_weight_keys is not None:
                scale_key = f"{name}.scale_weight"
                if scale_key in scale_weight_keys:
                    setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())
            
            original_forward = submodule.forward
            setattr(submodule, "original_forward", original_forward)
            setattr(submodule, "forward", 
                    lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))
            converted_count += 1
    
    print(f"   ✅ 已转换 {converted_count} 个 Linear 层为 FP8")
    print(f"   ⏭️  跳过 {skipped_count} 个特殊层（保持原精度）")


def load_fp8(pipe, repo_id, device, dtype):
    """FP8 量化加载
    
    Args:
        pipe: ModularPipeline 实例
        repo_id: 模型仓库 ID
        device: 目标设备
        dtype: 基础数据类型
    
    Returns:
        加载完成的 pipe
    """
    print("🔧 使用 FP8 优化 (基于 ComfyUI 实现)...")
    
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from diffusers import AutoModel
    
    # 检查硬件支持
    supports_fp8, compute_cap, msg = check_fp8_support()
    print(f"   GPU: {msg}")
    if not supports_fp8:
        raise RuntimeError(f"当前 GPU 不支持 FP8: {msg}")
    
    # 1. 下载 FP8 checkpoint
    fp8_repo = "6chan/krea-realtime-video-fp8"
    fp8_file = "krea-realtime-video-14b-fp8-e4m3fn.safetensors"
    print(f"   [1/4] 下载 FP8 权重: {fp8_repo}")
    
    fp8_path = hf_hub_download(repo_id=fp8_repo, filename=fp8_file)
    print(f"   ✅ 已下载: {fp8_path}")
    
    # 2. 加载其他组件（不包括 transformer）
    print("   [2/4] 加载其他组件 (VAE, Text Encoder)...")
    config_only_components = {"transformer", "guider", "video_processor", "scheduler"}
    specs = pipe._component_specs
    if isinstance(specs, dict):
        all_component_names = list(specs.keys())
    elif specs:
        first = next(iter(specs), None)
        if hasattr(first, 'name'):
            all_component_names = [spec.name for spec in specs]
        else:
            all_component_names = list(specs)
    else:
        all_component_names = []
    
    components_to_load = [name for name in all_component_names if name not in config_only_components]
    
    pipe.load_components(
        names=components_to_load,
        trust_remote_code=True,
        device_map=device,
        torch_dtype={"default": dtype, "vae": torch.float16},
    )
    
    # 3. 加载 FP8 权重
    print("   [3/4] 加载 FP8 权重...")
    fp8_state_dict = load_file(fp8_path)
    
    # 提取 scale_weights（去掉 model. 前缀）
    scale_weights = {}
    for k, v in fp8_state_dict.items():
        if k.endswith(".scale_weight") or k.endswith(".weight_scale"):
            normalized_k = k.replace(".weight_scale", ".scale_weight")
            if normalized_k.startswith("model."):
                normalized_k = normalized_k[6:]
            scale_weights[normalized_k] = v.to(device, torch.float32)
    
    # 需要保持原精度的层（不使用 FP8）
    params_to_keep = {
        "norm", "bias", "time_in", "patch_embedding", "time_", 
        "img_emb", "modulation", "text_embedding"
    }
    
    # 4. 加载 transformer 到 CPU，替换 FP8 权重，再移动到 GPU
    print("   [4/4] 加载 transformer 并替换 FP8 权重...")
    
    # 先加载到 CPU（避免 OOM）
    transformer = AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )
    
    # FP8 dtype 列表
    fp8_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
    
    # 获取模型的 state_dict keys 用于匹配
    model_state_dict = transformer.state_dict()
    
    # FP8 checkpoint 的 key 有 "model." 前缀，需要去掉
    def normalize_key(key):
        if key.startswith("model."):
            return key[6:]  # 去掉 "model." 前缀
        return key
    
    # 构建 normalized key -> original key 的映射
    fp8_key_map = {normalize_key(k): k for k in fp8_state_dict.keys()}
    
    # 只替换 FP8 权重（Linear 层的 weight）
    replaced_count = 0
    for model_key in model_state_dict.keys():
        # 检查 FP8 state_dict 中是否有对应的 key（去掉 model. 前缀后）
        if model_key in fp8_key_map:
            fp8_key = fp8_key_map[model_key]
            fp8_value = fp8_state_dict[fp8_key]
            
            # 只替换 FP8 格式的权重，且不在 params_to_keep 中
            is_fp8 = fp8_value.dtype in fp8_dtypes
            keep_original = any(keyword in model_key for keyword in params_to_keep)
            
            if is_fp8 and not keep_original:
                # 解析 key 找到模块
                parts = model_key.rsplit(".", 1)
                if len(parts) == 2:
                    module_name, param_name = parts
                    # 获取模块
                    module = transformer
                    for part in module_name.split("."):
                        module = getattr(module, part)
                    # 替换为 FP8 权重（先保持在 CPU）
                    setattr(module, param_name, 
                            nn.Parameter(fp8_value, requires_grad=False))
                    replaced_count += 1
    
    print(f"   ✅ 已替换 {replaced_count} 个权重为 FP8 格式")
    
    # 设置 scale_weights 到模块（在 CPU 上）
    for k, v in fp8_state_dict.items():
        if k.endswith(".scale_weight") or k.endswith(".weight_scale"):
            module_key = k.replace(".scale_weight", "").replace(".weight_scale", "")
            if module_key.startswith("model."):
                module_key = module_key[6:]
            parts = module_key.rsplit(".", 1)
            if len(parts) == 2:
                module_name = parts[0]
                try:
                    module = transformer
                    for part in module_name.split("."):
                        module = getattr(module, part)
                    setattr(module, "scale_weight", v.float())
                except AttributeError:
                    pass
    
    # 清理 CPU 内存
    del fp8_state_dict
    del model_state_dict
    import gc
    gc.collect()
    
    # 移动 transformer 到 GPU
    print("   📤 移动 transformer 到 GPU...")
    transformer = transformer.to(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"   ✅ 已移动到 GPU")
    
    pipe.transformer = transformer
    
    # 尝试融合投影层（在 FP8 优化之前，因为融合会创建新的 Linear 层）
    try:
        print("🔧 尝试融合投影层...")
        for block in pipe.transformer.blocks:
            block.self_attn.fuse_projections()
        print("   ✅ 融合成功")
    except Exception as e:
        print(f"   ⚠️  跳过 fuse_projections: {e}")
    
    # 应用 FP8 Linear 优化（在融合之后，这样可以包括新创建的融合层）
    convert_fp8_linear(pipe.transformer, dtype, params_to_keep, scale_weights)
    
    torch.cuda.empty_cache()
    print("   ✅ FP8 优化完成")
    
    # 设置 Text Encoder Offload 并立即执行（用于释放显存给 KV cache）
    from .offload import setup_text_encoder_offload, offload_text_encoder
    pipe = setup_text_encoder_offload(pipe)
    offload_text_encoder(pipe)  # 立即 offload，释放 ~10GB 显存
    
    return pipe
