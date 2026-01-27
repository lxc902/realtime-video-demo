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
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
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
# API Models
# ============================================================

class StreamGenerationRequest(BaseModel):
    prompt: str
    num_blocks: int = 5
    num_denoising_steps: int = NUM_INFERENCE_STEPS
    strength: float = DEFAULT_STRENGTH
    seed: Optional[int] = None
    start_frame: Optional[str] = None  # base64 encoded


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
        print(f"[SSE] Starting session {session_id}")
        
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
            
            # 生成每个 block
            for block_idx in range(req.num_blocks):
                def generate_block():
                    nonlocal state
                    with inference_lock:
                        input_frames = start_frames_list if block_idx == 0 and start_frames_list else None
                        new_state, frames = model.generate_next_block_with_state(
                            state=state,
                            prompt=req.prompt,
                            strength=req.strength,
                            block_idx=block_idx,
                            generator=generator,
                            input_frame=input_frames,
                            start_frame=None,
                            num_blocks=req.num_blocks
                        )
                        state = new_state
                        return frames
                
                frames = await asyncio.to_thread(generate_block)
                
                # 逐帧推送（模型层已保证每次返回 3 帧）
                for frame_idx, frame in enumerate(frames):
                    frame_bytes = model.frame_to_bytes(frame)
                    frame_b64 = base64.b64encode(frame_bytes).decode()
                    global_frame_idx = block_idx * len(frames) + frame_idx + 1
                    
                    event_data = json.dumps({
                        "type": "frame",
                        "block": block_idx,
                        "frame_idx": global_frame_idx,
                        "total_blocks": req.num_blocks,
                        "data": frame_b64
                    })
                    print(f"[SSE] {session_id}: frame {global_frame_idx}")
                    yield f"data: {event_data}\n\n"
            
            # 完成
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            print(f"[SSE] {session_id}: complete")
            
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
    uvicorn.run(app, host="0.0.0.0", port=7860)
