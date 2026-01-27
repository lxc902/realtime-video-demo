"""
KREA Realtime Video v2 - Streaming 版本
简化的后端，专注于 SSE 流式生成
"""
import os
import asyncio
import json
import base64
import uuid
import threading
import time
from datetime import datetime, timezone, timedelta

# 北京时间
def beijing_time():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S.%f")[:-3]
import gc
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# 导入本地推理模块
from local_inference import get_model
from config import (
    MODEL_PATH, QUANTIZATION,
    NUM_INFERENCE_STEPS, DEFAULT_STRENGTH,
    V2V_INITIAL_FRAMES
)

app = FastAPI(title="KREA Realtime Video v2")
templates = Jinja2Templates(directory=".")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSP Middleware
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if "text/html" in response.headers.get("content-type", ""):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self' wss: https:; "
                "img-src 'self' data: blob:; "
                "media-src 'self' blob:;"
            )
        return response

app.add_middleware(CSPMiddleware)

# 全局模型实例
model = None

# 推理锁 - 确保同一时间只有一个请求使用模型
inference_lock = threading.Lock()


def load_model_on_startup():
    """启动时加载模型"""
    global model
    print("")
    print("=" * 60)
    print("🔥 Loading KREA model to GPU...")
    if MODEL_PATH:
        print(f"   From: {MODEL_PATH}")
    else:
        print("   From: HuggingFace (krea/krea-realtime-video)")
    if QUANTIZATION:
        print(f"   Quantization: {QUANTIZATION.upper()}")
    else:
        print("   Quantization: None (full precision)")
    print("=" * 60)
    print("")
    model = get_model(model_path=MODEL_PATH, quantization=QUANTIZATION)
    print("")
    print("=" * 60)
    print("✅ Model loaded successfully!")
    print("🌐 Server ready at http://localhost:7860")
    print("=" * 60)
    print("")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_on_startup)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": "v2-streaming"
    }


# ============================================================
# 实时帧缓存（前端持续更新，后端生成时使用最新帧）
# ============================================================
latest_frame_lock = threading.Lock()
latest_frame_data = {
    "frame": None,  # 最新帧 (numpy array)
    "timestamp": 0,  # 服务器时间
    "client_ts": 0,  # 客户端时间戳（前端发送）
    "strength": None,  # 最新 strength
    "prompt": None     # 最新 prompt
}

# ============================================================
# API Models
# ============================================================

class StreamGenerationRequest(BaseModel):
    prompt: str
    num_blocks: int = 5  # 设为 0 表示无限生成
    num_denoising_steps: int = NUM_INFERENCE_STEPS
    strength: float = DEFAULT_STRENGTH
    seed: Optional[int] = None
    start_frame: Optional[str] = None  # base64 encoded（首帧，后续用 update_frame）

class UpdateFrameRequest(BaseModel):
    frame: Optional[str] = None  # base64 encoded (可选)
    timestamp: float = 0  # 客户端时间戳（ms）
    strength: Optional[float] = None
    prompt: Optional[str] = None


# ============================================================
# 帧更新 API（前端持续调用）
# ============================================================

@app.post("/api/update_frame")
async def api_update_frame(req: UpdateFrameRequest):
    """前端持续发送最新帧，后端缓存"""
    global latest_frame_data
    try:
        with latest_frame_lock:
            # 只有 frame 存在时才更新帧数据
            if req.frame:
                frame_bytes = base64.b64decode(req.frame)
                frame = model.process_frame_bytes(frame_bytes) if model else None
                latest_frame_data["frame"] = frame
                latest_frame_data["timestamp"] = time.time()
                latest_frame_data["client_ts"] = req.timestamp
            
            # strength/prompt 始终更新
            if req.strength is not None:
                latest_frame_data["strength"] = req.strength
            if req.prompt is not None:
                latest_frame_data["prompt"] = req.prompt
        
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# SSE 流式生成 API
# ============================================================

@app.post("/api/generate/stream")
async def api_stream_generation(req: StreamGenerationRequest):
    """SSE 流式生成 - 每生成一帧立即推送"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    async def generate_stream():
        session_id = str(uuid.uuid4())[:8]
        print(f"[{beijing_time()}] SSE session {session_id} started")
        
        try:
            # 处理起始帧
            start_frame = None
            start_frames_list = None
            if req.start_frame:
                start_frame_bytes = base64.b64decode(req.start_frame)
                start_frame = model.process_frame_bytes(start_frame_bytes)
                start_frames_list = [start_frame] * V2V_INITIAL_FRAMES
            
            # 初始化 state
            state = None
            generator = None
            
            def init_generation():
                nonlocal state, generator
                with inference_lock:
                    state, generator = model.initialize_generation_with_state(
                        prompt=req.prompt,
                        start_frame=start_frame,
                        num_inference_steps=req.num_denoising_steps,
                        strength=req.strength,
                        seed=req.seed
                    )
            
            await asyncio.to_thread(init_generation)
            
            # 生成循环（num_blocks=0 表示无限生成）
            block_idx = 0
            max_blocks = req.num_blocks if req.num_blocks > 0 else 999999
            global_start_time = time.time() * 1000  # 全局起始时间
            cumulative_time = 0  # 累计时间（用于帧时间戳）
            
            while block_idx < max_blocks:
                current_block = block_idx  # 闭包捕获
                block_start_time = time.time() * 1000  # ms
                
                def generate_block():
                    nonlocal state, start_frames_list
                    input_client_ts = 0  # 输入帧的客户端时间戳
                    with inference_lock:
                        # 获取最新帧、strength、prompt
                        current_prompt = req.prompt
                        current_strength = req.strength
                        with latest_frame_lock:
                            if latest_frame_data["frame"] is not None:
                                latest_frame = latest_frame_data["frame"]
                                start_frames_list = [latest_frame] * V2V_INITIAL_FRAMES
                                input_client_ts = latest_frame_data["client_ts"]
                            if latest_frame_data["strength"] is not None:
                                current_strength = latest_frame_data["strength"]
                            if latest_frame_data["prompt"] is not None:
                                current_prompt = latest_frame_data["prompt"]
                        
                        # 每个 block 都使用最新输入帧（减少延迟感）
                        input_frames = start_frames_list if start_frames_list else None
                        new_state, frames = model.generate_next_block_with_state(
                            state=state,
                            prompt=current_prompt,
                            strength=current_strength,
                            block_idx=current_block,
                            generator=generator,
                            input_frame=input_frames,
                            start_frame=None,
                            num_blocks=max_blocks
                        )
                        state = new_state
                        return frames, input_client_ts
                
                frames, input_ts = await asyncio.to_thread(generate_block)
                block_end_time = time.time() * 1000  # ms
                
                # 时间插值：将生成耗时均匀分配给每帧
                block_duration = block_end_time - block_start_time
                num_frames = len(frames)
                frame_interval = block_duration / num_frames  # 每帧间隔
                
                for frame_idx, frame in enumerate(frames):
                    frame_bytes = model.frame_to_bytes(frame)
                    frame_b64 = base64.b64encode(frame_bytes).decode()
                    global_frame_idx = block_idx * num_frames + frame_idx + 1
                    
                    # 相对时间戳（从 0 开始，累加）
                    # 前端可以直接用这个差值来计算播放间隔
                    frame_ts = cumulative_time + frame_interval * (frame_idx + 1)
                    
                    event_data = json.dumps({
                        "type": "frame",
                        "block": block_idx,
                        "frame_idx": global_frame_idx,
                        "timestamp": frame_ts,
                        "input_ts": input_ts,  # 输入帧的客户端时间戳（用于计算延迟）
                        "data": frame_b64
                    })
                    yield f"data: {event_data}\n\n"
                
                # 累加时间
                cumulative_time += block_duration
                print(f"[{beijing_time()}] Block {block_idx}: {num_frames} frames, {block_duration:.0f}ms")
                block_idx += 1
            
            # 完成（仅当 num_blocks > 0 时）
            if req.num_blocks > 0:
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                print(f"[{beijing_time()}] SSE session {session_id} complete")
            
            # 清理
            if model is not None and hasattr(model, 'cleanup_inference'):
                model.cleanup_inference()
            gc.collect()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/clear-cache")
async def clear_cache():
    """清理推理缓存"""
    if model is not None and hasattr(model, 'cleanup_inference'):
        model.cleanup_inference()
    gc.collect()
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        return {
            "status": "ok",
            "gpu_memory_gb": round(allocated, 2)
        }
    
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    # Ctrl+C 立即退出，不等待连接关闭
    def force_exit(sig, frame):
        print("\n强制退出...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)
    
    uvicorn.run(app, host="0.0.0.0", port=7860)
