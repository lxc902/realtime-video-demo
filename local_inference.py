"""
KREA Realtime Video - 本地 GPU 推理模块
使用 diffusers 库在本地 GPU 上运行 KREA 模型
"""
import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState
from PIL import Image
import numpy as np
import io


class KreaLocalInference:
    def __init__(self, device="cuda", dtype=torch.bfloat16, model_path=None, quantization=None):
        """初始化本地 KREA 模型
        
        Args:
            device: 设备 (cuda/cpu)
            dtype: 数据类型
            model_path: 自定义模型路径，可以是：
                       - 本地路径: "/path/to/model"
                       - HuggingFace repo: "krea/krea-realtime-video"
                       - None: 使用默认 HuggingFace repo
            quantization: 量化类型 (None, "fp8", "int8", "int4")
        """
        print("正在加载 KREA Realtime Video 模型...")
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        
        # 确定模型路径
        if model_path is None:
            repo_id = "krea/krea-realtime-video"
            print(f"从 HuggingFace 加载: {repo_id}")
        else:
            repo_id = model_path
            print(f"从自定义路径加载: {model_path}")
        
        # 加载 pipeline 结构
        self.pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
        
        # 使用量化模块加载模型
        from quantization import load_model_with_quantization
        
        try:
            self.pipe = load_model_with_quantization(
                self.pipe, repo_id, device, dtype, quantization
            )
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 检查关键组件
        if not hasattr(self.pipe, 'transformer') or self.pipe.transformer is None:
            raise RuntimeError(
                "❌ 模型加载失败: transformer 组件未正确加载\n"
                "可能原因:\n"
                "1. 缺少依赖包（einops, imageio, ftfy）\n"
                "2. diffusers 版本不兼容\n"
                "解决方法:\n"
                "  pip install einops imageio ftfy\n"
                "  然后重启服务"
            )
        
        print("模型加载完成！")
        
        self.state = None
        self.current_frames = []
        
    def initialize_generation(self, prompt, start_frame=None, num_inference_steps=4, strength=0.45, seed=None):
        """初始化生成过程"""
        # 彻底重置编译缓存，确保新生成使用正确的 block_mask
        import torch._dynamo
        torch._dynamo.reset()
        
        # 清理 inductor codecache
        try:
            import torch._inductor
            if hasattr(torch._inductor, 'codecache'):
                if hasattr(torch._inductor.codecache, 'PyCodeCache'):
                    torch._inductor.codecache.PyCodeCache.clear()
                if hasattr(torch._inductor.codecache, 'FxGraphCache'):
                    torch._inductor.codecache.FxGraphCache.clear()
        except Exception:
            pass
        
        # 重置 flex_attention 相关的编译缓存
        try:
            from torch.nn.attention.flex_attention import _flex_attention_cache
            if hasattr(_flex_attention_cache, 'clear'):
                _flex_attention_cache.clear()
        except Exception:
            pass
        
        # 重置 transformer 内部的 block_mask 缓存
        self._reset_transformer_caches()
        
        self.state = PipelineState()
        self.current_frames = []
        
        if seed is not None:
            self.generator = torch.Generator(self.device).manual_seed(seed)
        else:
            self.generator = None
            
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.start_frame = start_frame
        self.block_idx = 0
        
    def _cleanup_state_tensors(self):
        """清理 state 中的大张量，避免 deepcopy 时 OOM
        
        diffusers ModularPipeline.__call__ 会对 state 进行 deepcopy，
        如果 state 中累积了大量 GPU 张量，deepcopy 会导致 OOM。
        
        根本问题：kv_cache 和 crossattn_cache 各有 40 个元素，包含大量 GPU 张量。
        deepcopy 这些缓存需要额外 ~60GB 内存。
        
        解决方案：清理所有大缓存，让 pipeline 重新计算。牺牲速度换取内存稳定。
        """
        if self.state is None or not hasattr(self.state, 'values'):
            return
        
        values = self.state.values
        if not values:
            return
        
        # 强制清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 调试：打印 GPU 内存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"[Debug] block_idx={self.block_idx}, GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        # 删除所有大缓存（这些会导致 deepcopy OOM）
        keys_to_delete = [
            "videos",           # 生成的视频帧（已保存到 self.current_frames）
            "decoder_cache",    # VAE decoder 缓存（55个张量）
            "video_stream",     # 视频流输出
            "kv_cache",         # attention KV 缓存（40个元素，巨大！）
            "crossattn_cache",  # cross attention 缓存（40个元素，巨大！）
        ]
        
        deleted = []
        for key in keys_to_delete:
            if key in values:
                deleted.append(key)
                del values[key]
        
        if deleted:
            print(f"  [Cleanup] Deleted: {deleted}")
        
        # 再次清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 打印清理后的内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"  [After cleanup] GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    
    def generate_next_block(self, input_frame=None):
        """生成下一个 block 的帧"""
        # 清理 state 中的大张量，避免 deepcopy OOM
        self._cleanup_state_tensors()
        
        # 使用 inference_mode 比 no_grad 更激进，完全禁用 autograd 追踪
        with torch.inference_mode():
            kwargs = {
                "state": self.state,
                "prompt": [self.prompt],
                "num_inference_steps": self.num_inference_steps,
                "strength": self.strength,
                "block_idx": self.block_idx,
            }
            
            if self.generator is not None:
                kwargs["generator"] = self.generator
                
            # 如果是 video-to-video 或 webcam 模式，添加输入帧
            if input_frame is not None:
                kwargs["video"] = input_frame
            elif self.start_frame is not None and self.block_idx == 0:
                kwargs["video"] = self.start_frame
                
            # 生成
            self.state = self.pipe(**kwargs)
            
            # 提取生成的帧
            new_frames = self.state.values["videos"][0]
            self.current_frames.extend(new_frames)
            self.block_idx += 1
        
        # 显式删除临时变量
        del kwargs
        
        # 每帧推理后清理中间张量，防止内存累积
        torch.cuda.empty_cache()
        
        return new_frames
    
    def process_frame_bytes(self, frame_bytes):
        """将字节数据转换为模型可用的格式"""
        image = Image.open(io.BytesIO(frame_bytes))
        frame = np.array(image)
        return frame
    
    def frame_to_bytes(self, frame):
        """将帧转换为 JPEG 字节"""
        # 如果已经是 PIL Image，直接保存
        if isinstance(frame, Image.Image):
            buf = io.BytesIO()
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frame.save(buf, format='JPEG', quality=90)
            return buf.getvalue()
        
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        
        image = Image.fromarray(frame)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        return buf.getvalue()
    
    def _reset_transformer_caches(self):
        """重置 transformer 内部的 block_mask 和 kv cache"""
        if not hasattr(self.pipe, 'transformer') or self.pipe.transformer is None:
            return
        
        transformer = self.pipe.transformer
        cleared_count = 0
        
        # 遍历所有子模块，清除 block_mask 相关缓存
        for name, module in transformer.named_modules():
            # 清除 block_mask 相关属性
            attrs_to_clear = []
            for attr_name in dir(module):
                if 'block_mask' in attr_name.lower() or 'blockmask' in attr_name.lower():
                    attrs_to_clear.append(attr_name)
            
            for attr_name in attrs_to_clear:
                try:
                    if hasattr(module, attr_name):
                        setattr(module, attr_name, None)
                        cleared_count += 1
                except Exception:
                    pass
            
            # 清除 kv cache（如果存在）
            if hasattr(module, 'kv_cache'):
                module.kv_cache = None
            if hasattr(module, '_kv_cache'):
                module._kv_cache = None
            if hasattr(module, 'cache'):
                module.cache = None
        
        # 清除 transformer 级别的缓存
        if hasattr(transformer, 'block_mask'):
            transformer.block_mask = None
        if hasattr(transformer, '_block_mask'):
            transformer._block_mask = None
        if hasattr(transformer, 'kv_cache'):
            transformer.kv_cache = None
        if hasattr(transformer, 'reset_caches'):
            try:
                transformer.reset_caches()
                cleared_count += 1
            except Exception:
                pass
        
        if cleared_count > 0:
            print(f"[Cache Reset] Cleared {cleared_count} transformer caches")
    
    def cleanup_inference(self):
        """清理推理过程中的临时显存"""
        # 彻底重置 torch.compile 和 inductor 缓存
        import torch._dynamo
        torch._dynamo.reset()
        
        # 清理 inductor codecache（更彻底的清理）
        try:
            import torch._inductor
            if hasattr(torch._inductor, 'codecache'):
                if hasattr(torch._inductor.codecache, 'PyCodeCache'):
                    torch._inductor.codecache.PyCodeCache.clear()
                if hasattr(torch._inductor.codecache, 'FxGraphCache'):
                    torch._inductor.codecache.FxGraphCache.clear()
        except Exception:
            pass
        
        # 重置 flex_attention 相关的编译缓存
        try:
            from torch.nn.attention.flex_attention import _flex_attention_cache
            if hasattr(_flex_attention_cache, 'clear'):
                _flex_attention_cache.clear()
        except Exception:
            pass
        
        # 重置 transformer 内部的 block_mask 缓存
        self._reset_transformer_caches()
        
        # 清理状态
        if self.state is not None:
            # 清理 state 中的 tensors
            if hasattr(self.state, 'values') and self.state.values:
                self.state.values.clear()
            self.state = None
        
        # 清理帧列表
        self.current_frames.clear()
        
        # 清理 generator
        self.generator = None
        
        # 强制 CUDA 垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()


# 单例模式
_model_instance = None


def get_model(model_path=None, quantization=None):
    """获取模型单例
    
    Args:
        model_path: 自定义模型路径 (可选)
        quantization: 量化类型 (可选)
                     - None: 不量化 (~54GB+ 显存)
                     - "fp8": FP8 量化 (~24GB 显存)
                     - "int8": INT8 量化 (~28GB 显存)
                     - "int4": INT4 量化 (~16GB 显存)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = KreaLocalInference(model_path=model_path, quantization=quantization)
    return _model_instance
