"""
BF16 标准加载（无量化）
需要 ~54GB+ 显存
"""
import torch


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
    
    pipe.load_components(
        trust_remote_code=True,
        device_map=device,
        torch_dtype={"default": dtype, "vae": torch.float16},
    )
    
    # 融合投影层优化
    print("🔧 融合投影层...")
    for block in pipe.transformer.blocks:
        block.self_attn.fuse_projections()
    
    print("   ✅ BF16 加载完成")
    return pipe
