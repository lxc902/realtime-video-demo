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
    def __init__(self, device="cuda", dtype=torch.bfloat16, model_path=None):
        """初始化本地 KREA 模型
        
        Args:
            device: 设备 (cuda/cpu)
            dtype: 数据类型
            model_path: 自定义模型路径，可以是：
                       - 本地路径: "/path/to/model"
                       - HuggingFace repo: "krea/krea-realtime-video"
                       - None: 使用默认 HuggingFace repo
        """
        print("正在加载 KREA Realtime Video 模型...")
        self.device = device
        self.dtype = dtype
        
        # 确定模型路径
        if model_path is None:
            # 默认使用 HuggingFace
            repo_id = "krea/krea-realtime-video"
            print(f"从 HuggingFace 加载: {repo_id}")
        else:
            # 使用自定义路径
            repo_id = model_path
            print(f"从自定义路径加载: {model_path}")
        
        # 加载模型
        self.pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
        self.pipe.load_components(
            trust_remote_code=True,
            device_map=device,
            torch_dtype={"default": dtype, "vae": torch.float16},
        )
        
        # 优化: 融合投影层
        if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
            for block in self.pipe.transformer.blocks:
                block.self_attn.fuse_projections()
        else:
            print("警告: transformer 未正确加载，性能可能受影响")
        
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

def get_model(model_path=None):
    """获取模型单例
    
    Args:
        model_path: 自定义模型路径 (可选)
                   - 本地路径: "/path/to/model"
                   - HuggingFace repo: "krea/krea-realtime-video"
                   - None: 使用默认
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = KreaLocalInference(model_path=model_path)
    return _model_instance
