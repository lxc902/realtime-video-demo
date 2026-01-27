"""
BF16 标准加载（无量化）
需要 ~54GB+ 显存
"""
import torch
import os


def load_bf16(pipe, repo_id, device, dtype):
    """BF16 标准加载模型（无量化）
    
    Args:
        pipe: ModularPipeline 实例
        repo_id: 模型仓库 ID
        device: 目标设备
        dtype: 数据类型 (默认 bfloat16)
    
    Returns:
        加载完成的 pipe
    """
    print("🔧 BF16 标准加载（无量化）...")
    print("   ⚠️  需要 ~54GB+ 显存")
    
    # CUDA 性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("   ✅ CUDA 优化已启用 (cudnn.benchmark, TF32)")
    
    pipe.load_components(
        trust_remote_code=True,
        device_map=device,
        torch_dtype={"default": dtype, "vae": torch.float16},
    )
    
    # 融合投影层优化
    print("🔧 融合投影层...")
    for block in pipe.transformer.blocks:
        block.self_attn.fuse_projections()
    
    # torch.compile 优化
    # 注意：KREA 模型的动态缓存与 CUDAGraphs 不兼容
    # 使用 mode="default" 而非 "reduce-overhead"
    if os.environ.get("DISABLE_COMPILE", "0") != "1":
        try:
            print("🔧 编译 transformer (torch.compile, mode=default)...")
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="default",  # 不使用 CUDAGraphs，避免缓存冲突
                fullgraph=False,
            )
            print("   ✅ torch.compile 完成（首次推理会较慢）")
        except Exception as e:
            print(f"   ⚠️  torch.compile 跳过: {e}")
    
    print("   ✅ BF16 加载完成")
    return pipe
