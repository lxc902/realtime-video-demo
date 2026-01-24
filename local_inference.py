"""
KREA Realtime Video - 本地 GPU 推理模块
使用 diffusers 库在本地 GPU 上运行 KREA 模型
"""
import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState
from diffusers.utils import load_video, export_to_video
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
            quantization: 量化类型 (None, "int8", "int4")
        """
        print("正在加载 KREA Realtime Video 模型...")
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        
        # 确定模型路径
        if model_path is None:
            # 总是从原始 repo 加载 pipeline 结构
            repo_id = "krea/krea-realtime-video"
            print(f"从 HuggingFace 加载: {repo_id}")
        else:
            # 使用自定义路径
            repo_id = model_path
            print(f"从自定义路径加载: {model_path}")
        
        # 加载模型
        self.pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
        
        # 根据量化类型加载模型
        if quantization == "fp8":
            # FP8 预量化模型
            print("🔧 使用 FP8 预量化模型 (预计显存 ~24GB)")
            try:
                from huggingface_hub import hf_hub_download
                
                # 1. 下载 FP8 transformer checkpoint
                fp8_repo = "6chan/krea-realtime-video-fp8"
                fp8_file = "krea-realtime-video-14b-fp8-e4m3fn.safetensors"
                print(f"   [1/3] 下载 FP8 transformer: {fp8_repo}")
                
                fp8_path = hf_hub_download(
                    repo_id=fp8_repo,
                    filename=fp8_file,
                )
                print(f"   ✅ FP8 checkpoint: {fp8_path}")
                
                # 2. 加载其他组件（不包括 transformer）
                print("   [2/3] 加载其他组件...")
                config_only_components = {"transformer", "guider", "video_processor", "scheduler"}
                specs = self.pipe._component_specs
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
                
                self.pipe.load_components(
                    names=components_to_load,
                    trust_remote_code=True,
                    device_map=device,
                    torch_dtype={"default": dtype, "vae": torch.float16},
                )
                
                # 3. 加载 FP8 transformer
                print("   [3/3] 加载 FP8 transformer...")
                from safetensors.torch import load_file
                
                # 先加载原始 transformer 结构
                from diffusers import AutoModel
                transformer = AutoModel.from_pretrained(
                    repo_id,
                    subfolder="transformer",
                    torch_dtype=torch.float8_e4m3fn,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                # 加载 FP8 权重
                fp8_state_dict = load_file(fp8_path)
                transformer.load_state_dict(fp8_state_dict, strict=False)
                transformer = transformer.to(device)
                
                self.pipe.transformer = transformer
                
                torch.cuda.empty_cache()
                print("   ✅ FP8 模型加载完成")
                
            except Exception as e:
                print(f"   ❌ FP8 加载失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"FP8 加载失败: {e}")
        elif quantization in ("int8", "int4"):
            # 使用 torchao 量化（替代 bitsandbytes，兼容性更好）
            print(f"🔧 启用 {quantization.upper()} 量化 (torchao)...")
            
            try:
                from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only
                
                # 1. 先标准加载所有组件
                print("   [1/3] 正在加载模型组件...")
                self.pipe.load_components(
                    trust_remote_code=True,
                    device_map=device,
                    torch_dtype={"default": dtype, "vae": torch.float16},
                )
                
                # 2. 定义量化过滤器：只量化 Linear 层，跳过 Conv2D
                def linear_only_filter(module, name):
                    return isinstance(module, torch.nn.Linear)
                
                # 3. 对 transformer 进行量化
                print("   [2/3] 正在量化 transformer (仅 Linear 层)...")
                if quantization == "int8":
                    print("   使用 INT8 动态量化 (预计显存 ~28GB)")
                    quantize_(
                        self.pipe.transformer, 
                        int8_dynamic_activation_int8_weight(),
                        filter_fn=linear_only_filter
                    )
                else:  # int4
                    print("   使用 INT4 权重量化 (预计显存 ~16GB)")
                    quantize_(
                        self.pipe.transformer,
                        int4_weight_only(),
                        filter_fn=linear_only_filter
                    )
                
                # 4. 清理显存
                print("   [3/3] 清理显存缓存...")
                torch.cuda.empty_cache()
                print("   ✅ torchao 量化完成")
                
            except ImportError as e:
                print(f"   ❌ 量化失败: {e}")
                print("   请安装 torchao: pip install torchao")
                raise RuntimeError(f"量化失败，缺少依赖: {e}")
            except Exception as e:
                print(f"   ❌ 量化失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"量化失败: {e}")
        else:
            # 标准加载（无量化）
            self.pipe.load_components(
                trust_remote_code=True,
                device_map=device,
                torch_dtype={"default": dtype, "vae": torch.float16},
            )
        
        # 检查关键组件是否加载成功
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
        
        # 优化: 融合投影层（量化模式下跳过，因为可能不兼容）
        if not quantization:
            print("🔧 融合投影层...")
            for block in self.pipe.transformer.blocks:
                block.self_attn.fuse_projections()
        elif quantization == "fp8":
            # FP8 可以尝试 fuse_projections，但如果失败就跳过
            try:
                print("🔧 尝试融合投影层...")
                for block in self.pipe.transformer.blocks:
                    block.self_attn.fuse_projections()
                print("   ✅ 融合成功")
            except Exception as e:
                print(f"   ⚠️  跳过 fuse_projections: {e}")
        else:
            print("⚠️  量化模式下跳过 fuse_projections（不兼容）")
        
        print("模型加载完成！")
        
        self.state = None
        self.current_frames = []
        
    def initialize_generation(self, prompt, start_frame=None, num_inference_steps=4, strength=0.45, seed=None):
        """初始化生成过程"""
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
        
    def generate_next_block(self, input_frame=None):
        """生成下一个 block 的帧"""
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
        
        return new_frames
    
    def process_frame_bytes(self, frame_bytes):
        """将字节数据转换为模型可用的格式"""
        # 将 JPEG 字节转换为 PIL Image
        image = Image.open(io.BytesIO(frame_bytes))
        # 转换为 numpy array
        frame = np.array(image)
        return frame
    
    def frame_to_bytes(self, frame):
        """将帧转换为 JPEG 字节"""
        if isinstance(frame, torch.Tensor):
            # Tensor -> numpy
            frame = frame.cpu().numpy()
            # 假设范围是 [-1, 1] 或 [0, 1]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        
        # numpy -> PIL -> bytes
        image = Image.fromarray(frame)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        return buf.getvalue()


# 单例模式 - 避免重复加载模型
_model_instance = None

def get_model(model_path=None, quantization=None):
    """获取模型单例
    
    Args:
        model_path: 自定义模型路径 (可选)
                   - 本地路径: "/path/to/model"
                   - HuggingFace repo: "krea/krea-realtime-video"
                   - None: 使用默认
        quantization: 量化类型 (可选)
                     - None: 不量化 (需要 ~54GB+ 显存)
                     - "int8": 8位量化 (需要 ~24GB 显存)
                     - "int4": 4位量化 (需要 ~12GB 显存)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = KreaLocalInference(model_path=model_path, quantization=quantization)
    return _model_instance
