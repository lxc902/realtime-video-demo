"""
INT8 量化加载
使用 torchao 库进行动态量化
需要 ~28GB 显存
"""
import torch
import gc


def load_int8(pipe, repo_id, device, dtype):
    """INT8 量化加载
    
    Args:
        pipe: ModularPipeline 实例
        repo_id: 模型仓库 ID
        device: 目标设备
        dtype: 基础数据类型
    
    Returns:
        加载完成的 pipe
    """
    print("🔧 启用 INT8 量化 (torchao)...")
    
    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
    except ImportError as e:
        print("   ❌ 缺少 torchao 依赖")
        print("   请安装: pip install torchao")
        raise RuntimeError(f"INT8 量化失败，缺少依赖: {e}")
    
    # 1. 标准加载所有组件
    print("   [1/3] 正在加载模型组件...")
    pipe.load_components(
        trust_remote_code=True,
        device_map=device,
        torch_dtype={"default": dtype, "vae": torch.float16},
    )
    
    # 2. 定义量化过滤器：只量化 Linear 层
    def linear_only_filter(module, name):
        return isinstance(module, torch.nn.Linear)
    
    # 3. 对 transformer 进行量化
    print("   [2/3] 正在量化 transformer (仅 Linear 层)...")
    print("   使用 INT8 动态量化 (预计显存 ~28GB)")
    
    quantize_(
        pipe.transformer, 
        int8_dynamic_activation_int8_weight(),
        filter_fn=linear_only_filter
    )
    
    # 4. 清理显存和 CPU 内存
    print("   [3/3] 清理显存缓存...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 量化模式下跳过 fuse_projections
    print("   ⚠️  量化模式下跳过 fuse_projections（不兼容）")
    
    print("   ✅ INT8 量化完成")
    
    # 设置 Text Encoder Offload 并立即执行（用于释放显存给 KV cache）
    from .offload import setup_text_encoder_offload, offload_text_encoder
    pipe = setup_text_encoder_offload(pipe)
    offload_text_encoder(pipe)  # 立即 offload，释放 ~10GB 显存
    
    return pipe
