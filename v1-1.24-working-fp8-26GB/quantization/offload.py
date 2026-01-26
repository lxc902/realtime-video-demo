"""
Text Encoder Offload 辅助模块

用于在量化模式下将 Text Encoder 卸载到 CPU，释放 GPU 显存给 KV cache。
Text Encoder (T5-XXL) 约占用 10-12GB 显存。
"""
import torch
import gc


class TextEncoderOffloadHelper:
    """Text Encoder Offload 管理器
    
    在量化模式下，模型加载后 Text Encoder 仍占用约 10GB 显存。
    此类提供方法在 prompt 编码完成后将 Text Encoder 卸载到 CPU，
    并在需要编码新 prompt 时临时移回 GPU。
    """
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.text_encoder_on_cpu = False
        self.original_device = None
        
        # 查找 Text Encoder 组件
        self.text_encoder = None
        self.text_encoder_name = None
        
        # KREA 模型可能使用不同的 Text Encoder 名称
        possible_names = ['text_encoder', 'text_encoder_1', 'text_encoder_2', 't5_encoder']
        for name in possible_names:
            if hasattr(pipe, name) and getattr(pipe, name) is not None:
                self.text_encoder = getattr(pipe, name)
                self.text_encoder_name = name
                break
        
        if self.text_encoder is not None:
            # 获取原始设备
            try:
                first_param = next(self.text_encoder.parameters())
                self.original_device = first_param.device
            except StopIteration:
                self.original_device = torch.device('cuda')
    
    def offload_to_cpu(self):
        """将 Text Encoder 卸载到 CPU
        
        应在 prompt 编码完成后调用（通常是第一个 block 生成后）
        """
        if self.text_encoder is None:
            return
        
        if self.text_encoder_on_cpu:
            return  # 已经在 CPU 上
        
        print(f"📤 Offloading {self.text_encoder_name} to CPU...")
        
        # 移动到 CPU
        self.text_encoder.to('cpu')
        self.text_encoder_on_cpu = True
        
        # 清理 GPU 缓存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 打印释放的显存
        allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        print(f"   ✅ Done. GPU memory now: {allocated:.2f}GB")
    
    def restore_to_gpu(self):
        """将 Text Encoder 恢复到 GPU
        
        应在需要编码新 prompt 前调用
        """
        if self.text_encoder is None:
            return
        
        if not self.text_encoder_on_cpu:
            return  # 已经在 GPU 上
        
        print(f"📥 Restoring {self.text_encoder_name} to GPU...")
        
        device = self.original_device or torch.device('cuda')
        self.text_encoder.to(device)
        self.text_encoder_on_cpu = False
        
        torch.cuda.synchronize()
        print(f"   ✅ Done")
    
    def encode_prompt_with_offload(self, encode_fn, *args, **kwargs):
        """编码 prompt 并自动处理 offload
        
        如果 Text Encoder 在 CPU 上，临时移到 GPU，编码后移回 CPU。
        
        Args:
            encode_fn: 编码函数
            *args, **kwargs: 传递给编码函数的参数
        
        Returns:
            编码结果
        """
        was_on_cpu = self.text_encoder_on_cpu
        
        if was_on_cpu:
            self.restore_to_gpu()
        
        try:
            result = encode_fn(*args, **kwargs)
        finally:
            if was_on_cpu:
                self.offload_to_cpu()
        
        return result


def setup_text_encoder_offload(pipe):
    """为 pipeline 设置 Text Encoder Offload
    
    在量化模式下调用，将 offload helper 附加到 pipeline。
    
    Args:
        pipe: ModularPipeline 实例
    
    Returns:
        pipe (带有 offload_helper 属性)
    """
    helper = TextEncoderOffloadHelper(pipe)
    pipe._text_encoder_offload_helper = helper
    
    if helper.text_encoder is not None:
        print(f"🔧 Text Encoder Offload 已启用 ({helper.text_encoder_name})")
    else:
        print("⚠️  未找到 Text Encoder，跳过 Offload 设置")
    
    return pipe


def offload_text_encoder(pipe):
    """卸载 Text Encoder 到 CPU（便捷函数）"""
    if hasattr(pipe, '_text_encoder_offload_helper'):
        pipe._text_encoder_offload_helper.offload_to_cpu()


def restore_text_encoder(pipe):
    """恢复 Text Encoder 到 GPU（便捷函数）"""
    if hasattr(pipe, '_text_encoder_offload_helper'):
        pipe._text_encoder_offload_helper.restore_to_gpu()
