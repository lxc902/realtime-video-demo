"""
KREA Realtime Video - 本地 GPU 推理模块
使用 diffusers 库在本地 GPU 上运行 KREA 模型
"""
import torch
import torch._dynamo
import torch._inductor
import gc
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
        """初始化生成过程（旧 API，保留兼容性）"""
        state, generator = self.initialize_generation_with_state(
            prompt, start_frame, num_inference_steps, strength, seed
        )
        self.state = state
        self.generator = generator
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.start_frame = start_frame
        self.block_idx = 0
        self.current_frames = []
    
    def initialize_generation_with_state(self, prompt, start_frame=None, num_inference_steps=4, strength=0.45, seed=None):
        """初始化生成过程，返回独立的 state（新 API，支持多 session）"""
        # 彻底重置编译缓存，确保新生成使用正确的 block_mask
        torch._dynamo.reset()
        
        # 清理 inductor codecache
        try:
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
        
        # 重置 transformer 内部的所有缓存（包括 block_mask）
        self._reset_transformer_caches()
        
        # 彻底清理 pipeline 内部的所有缓存，确保新生成完全独立
        self._reset_pipeline_caches()
        
        # 创建新的独立 state
        state = PipelineState()
        
        # 创建 generator
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None
        
        # 保存常用参数到实例（用于旧 API 兼容）
        self.num_inference_steps = num_inference_steps
        
        print(f"[New Session] All caches cleared, starting fresh generation")
        
        return state, generator
        
    def _cleanup_state_tensors(self):
        """清理 state 中的大张量（旧 API，保留兼容性）"""
        self._cleanup_state_tensors_for_state(self.state, self.block_idx)
    
    def _cleanup_state_tensors_for_state(self, state, block_idx):
        """清理 state 中的大张量，避免 deepcopy 时 OOM
        
        diffusers ModularPipeline.__call__ 会对 state 进行 deepcopy，
        如果 state 中累积了大量 GPU 张量，deepcopy 会导致 OOM。
        """
        # 强制 Python 垃圾回收
        gc.collect()
        
        # 强制清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 调试：打印 GPU 内存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"[Debug] block_idx={block_idx}, GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        if state is None or not hasattr(state, 'values'):
            print(f"  [Cleanup] state is None or has no values")
            return
        
        values = state.values
        if not values:
            print(f"  [Cleanup] state.values is empty")
            return
        
        # 诊断：打印 state.values 中的所有 key
        all_keys = list(values.keys())
        print(f"  [Cleanup] state.values keys: {all_keys}")
        
        # 关键诊断：检查 current_denoised_latents 的状态
        if 'current_denoised_latents' in values:
            cdl = values['current_denoised_latents']
            if cdl is None:
                print(f"  [WARNING] current_denoised_latents is None!")
            elif hasattr(cdl, 'shape'):
                print(f"  [Debug] current_denoised_latents shape: {cdl.shape}")
            else:
                print(f"  [Debug] current_denoised_latents type: {type(cdl)}")
        else:
            print(f"  [WARNING] current_denoised_latents not in state.values!")
        
        # 暂时禁用删除，测试推理是否正常
        keys_to_delete = []
        
        deleted = []
        for key in keys_to_delete:
            if key in values:
                deleted.append(key)
                del values[key]
        
        if deleted:
            print(f"  [Cleanup] Deleted: {deleted}")
        else:
            print(f"  [Cleanup] Nothing to delete from target keys")
        
        # 强制 Python 垃圾回收
        gc.collect()
        
        # 再次清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 打印清理后的内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"  [After cleanup] GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    
    def generate_next_block(self, input_frame=None, num_blocks=25):
        """生成下一个 block 的帧（旧 API，保留兼容性）"""
        new_state, frames = self.generate_next_block_with_state(
            state=self.state,
            prompt=self.prompt,
            strength=self.strength,
            block_idx=self.block_idx,
            generator=self.generator,
            input_frame=input_frame,
            start_frame=self.start_frame,
            num_blocks=num_blocks
        )
        self.state = new_state
        self.block_idx += 1
        self.current_frames.extend(frames)
        return frames
    
    def generate_next_block_with_state(self, state, prompt, strength, block_idx, generator=None, input_frame=None, start_frame=None, num_blocks=25):
        """生成下一个 block 的帧，使用传入的 state（新 API，支持多 session）
        
        返回: (new_state, frames)
        """
        print(f"\n[generate_next_block] block_idx={block_idx}, input_frame={'provided' if input_frame is not None else 'None'}")
        
        # 在清理前检查 current_denoised_latents
        if state is not None and hasattr(state, 'values') and state.values:
            if 'current_denoised_latents' in state.values:
                cdl = state.values['current_denoised_latents']
                if cdl is None:
                    print(f"  [BEFORE cleanup] current_denoised_latents is None!")
                elif hasattr(cdl, 'shape'):
                    print(f"  [BEFORE cleanup] current_denoised_latents shape: {cdl.shape}")
            else:
                print(f"  [BEFORE cleanup] current_denoised_latents not in state.values")
        else:
            print(f"  [BEFORE cleanup] state is empty or None")
        
        # 清理 state 中的大张量，避免 deepcopy OOM
        self._cleanup_state_tensors_for_state(state, block_idx)
        
        # 使用 inference_mode 比 no_grad 更激进，完全禁用 autograd 追踪
        with torch.inference_mode():
            kwargs = {
                "state": state,
                "prompt": [prompt],
                "num_inference_steps": self.num_inference_steps,
                "strength": strength,
                "block_idx": block_idx,
                "num_blocks": num_blocks,  # 关键：必须传 num_blocks
            }
            
            if generator is not None:
                kwargs["generator"] = generator
             
            # 根据 KREA 官方代码 (before_denoise.py)：
            # - T2V 模式：不需要 video/video_stream 参数
            # - V2V 模式：使用 video 参数（整个预录视频）
            # - Streaming V2V 模式：使用 video_stream 参数（PIL Image 列表）
            #
            # 关键问题：
            # 1. Streaming 模式的 VAE temporal compression 需要足够多的帧
            # 2. 官方示例每次传入 12 帧
            # 3. 如果帧不够，编码后的 latents 帧数 < num_frames_per_block，会报错
            #
            # 解决方案：确保传入足够的帧到 video_stream
            
            has_input = (input_frame is not None) or (start_frame is not None)
            
            if has_input:
                # Streaming V2V 模式：使用 video_stream
                # video_stream 期望 PIL Image 列表
                from PIL import Image
                import numpy as np
                
                frame_to_use = input_frame if input_frame is not None else start_frame
                
                # 转换为 PIL Image 列表
                pil_list = []
                if isinstance(frame_to_use, list):
                    for f in frame_to_use:
                        if isinstance(f, np.ndarray):
                            pil_list.append(Image.fromarray(f))
                        elif isinstance(f, Image.Image):
                            pil_list.append(f)
                        else:
                            pil_list.append(f)
                elif isinstance(frame_to_use, np.ndarray):
                    pil_list.append(Image.fromarray(frame_to_use))
                elif isinstance(frame_to_use, Image.Image):
                    pil_list.append(frame_to_use)
                else:
                    pil_list.append(frame_to_use)
                
                print(f"  [video_stream] passing {len(pil_list)} frames to pipeline")
                kwargs["video_stream"] = pil_list
                
            # 生成
            new_state = self.pipe(**kwargs)
            
            # 诊断：检查 pipeline 执行后的 current_denoised_latents
            if hasattr(new_state, 'values') and 'current_denoised_latents' in new_state.values:
                cdl = new_state.values['current_denoised_latents']
                if cdl is None:
                    print(f"  [After pipe] current_denoised_latents is None!")
                elif hasattr(cdl, 'shape'):
                    print(f"  [After pipe] current_denoised_latents shape: {cdl.shape}")
            
            # 提取生成的帧
            new_frames = new_state.values["videos"][0]
        
        # 显式删除临时变量
        del kwargs
        
        # 强制 Python 垃圾回收
        gc.collect()
        
        # 每帧推理后清理中间张量，防止内存累积
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return new_state, new_frames
    
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
        """重置 transformer 内部的 block_mask 和所有缓存
        
        在新 session 开始时调用，确保完全独立的生成
        """
        if not hasattr(self.pipe, 'transformer') or self.pipe.transformer is None:
            return
        
        transformer = self.pipe.transformer
        cleared_count = 0
        
        # 清理 block_mask 相关属性
        for name, module in transformer.named_modules():
            for attr_name in ['block_mask', '_block_mask', 'blockmask', '_blockmask']:
                if hasattr(module, attr_name):
                    try:
                        val = getattr(module, attr_name)
                        if val is not None and not callable(val):
                            setattr(module, attr_name, None)
                            cleared_count += 1
                    except Exception:
                        pass
        
        # 清除 transformer 级别的 block_mask
        if hasattr(transformer, 'block_mask'):
            transformer.block_mask = None
            cleared_count += 1
        if hasattr(transformer, '_block_mask'):
            transformer._block_mask = None
            cleared_count += 1
        
        # 调用 transformer 的 reset_caches（清理 kv_cache 等）
        if hasattr(transformer, 'reset_caches') and callable(transformer.reset_caches):
            try:
                transformer.reset_caches()
                cleared_count += 1
                print(f"[Cache Reset] Called transformer.reset_caches()")
            except Exception as e:
                print(f"[Cache Reset] transformer.reset_caches() failed: {e}")
        
        if cleared_count > 0:
            print(f"[Cache Reset] Cleared {cleared_count} transformer caches")
    
    def _reset_pipeline_caches(self):
        """重置 pipeline 内部的所有缓存，确保新生成完全独立"""
        cleared = []
        
        # 清理 pipeline blocks 中的缓存
        if hasattr(self.pipe, 'blocks'):
            for block in self.pipe.blocks:
                # 清理 input_frames_cache（KREA 用于累积输入帧）
                if hasattr(block, 'input_frames_cache'):
                    block.input_frames_cache = None
                    cleared.append('input_frames_cache')
                
                # 清理 frame_cache_context
                if hasattr(block, 'frame_cache_context'):
                    block.frame_cache_context = None
                    cleared.append('frame_cache_context')
                
                # 清理 decoder_cache
                if hasattr(block, 'decoder_cache'):
                    block.decoder_cache = None
                    cleared.append('decoder_cache')
                
                # 清理 init_latents
                if hasattr(block, 'init_latents'):
                    block.init_latents = None
                    cleared.append('init_latents')
        
        # 清理 pipeline 级别的缓存
        cache_attrs = [
            'input_frames_cache', 'frame_cache_context', 'decoder_cache',
            'init_latents', 'kv_cache', 'crossattn_cache', '_cached_latents'
        ]
        for attr in cache_attrs:
            if hasattr(self.pipe, attr):
                setattr(self.pipe, attr, None)
                cleared.append(f'pipe.{attr}')
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if cleared:
            print(f"[Pipeline Reset] Cleared: {list(set(cleared))}")
    
    def cleanup_inference(self):
        """清理推理过程中的临时显存"""
        # 彻底重置 torch.compile 和 inductor 缓存
        torch._dynamo.reset()
        
        # 清理 inductor codecache（更彻底的清理）
        try:
            if hasattr(torch._inductor, 'codecache'):
                if hasattr(torch._inductor.codecache, 'PyCodeCache'):
                    torch._inductor.codecache.PyCodeCache.clear()
                if hasattr(torch._inductor.codecache, 'FxGraphCache'):
                    torch._inductor.codecache.FxGraphCache.clear()
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
