"""
KREA Realtime Video - 本地 GPU 版本
使用本地 GPU 而不是 FAL API
"""
import os
import asyncio
import json
from typing import Optional
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import msgpack

# 导入本地推理模块
from local_inference import get_model
from config import MODEL_PATH, QUANTIZATION

app = FastAPI()
templates = Jinja2Templates(directory=".")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CSP middleware
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

# Track active WebSocket connections
active_websockets = set()

# 全局模型实例
model = None

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
    print("   This will take 1-2 minutes on first run")
    print("=" * 60)
    print("")
    model = get_model(model_path=MODEL_PATH, quantization=QUANTIZATION)
    print("")
    print("=" * 60)
    print("✅ Model loaded successfully!")
    print("🌐 Server is ready to accept connections")
    print("=" * 60)
    print("")

async def cleanup_expired_sessions():
    """后台任务：定期清理超时的 HTTP sessions"""
    while True:
        await asyncio.sleep(30)  # 每 30 秒检查一次
        
        expired_sessions = []
        with session_lock:
            for session_id, session in list(active_sessions.items()):
                if session.is_expired():
                    expired_sessions.append((session_id, session))
        
        # 清理超时的 sessions
        for session_id, session in expired_sessions:
            with session_lock:
                if session_id in active_sessions:
                    # 清理推理显存
                    if hasattr(session.model, 'cleanup_inference'):
                        session.model.cleanup_inference()
                    del active_sessions[session_id]
                    print(f"[Cleanup] Session {session_id} expired and cleaned (timeout: {SESSION_TIMEOUT}s)")

@app.on_event("startup")
async def startup_event():
    """应用启动时的事件"""
    import asyncio
    # 在后台线程加载模型，避免阻塞启动
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_on_startup)
    
    # 启动后台清理任务
    asyncio.create_task(cleanup_expired_sessions())


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "mode": "local_gpu",
        "model_loaded": model is not None,
        "ready": model is not None
    }


@app.post("/api/clear-cache")
async def clear_inference_cache():
    """清理推理缓存（不影响模型加载）"""
    import gc
    
    cleaned_sessions = 0
    
    # 清理所有活跃的 HTTP sessions
    with session_lock:
        for session_id, session in list(active_sessions.items()):
            if hasattr(session.model, 'cleanup_inference'):
                session.model.cleanup_inference()
            del active_sessions[session_id]
            cleaned_sessions += 1
    
    # 清理全局模型的推理状态
    if model is not None and hasattr(model, 'cleanup_inference'):
        model.cleanup_inference()
    
    # 强制垃圾回收
    gc.collect()
    
    # 获取当前显存状态
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Clear Cache] Sessions cleaned: {cleaned_sessions}, GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return {
            "status": "ok",
            "sessions_cleaned": cleaned_sessions,
            "gpu_memory_allocated_gb": round(allocated, 2),
            "gpu_memory_reserved_gb": round(reserved, 2)
        }
    
    return {
        "status": "ok",
        "sessions_cleaned": cleaned_sessions
    }


# ============================================================
# RESTful API（HTTP 轮询模式，替代 WebSocket）
# ============================================================
import base64
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import uuid
import threading
import time

# 存储活跃的生成会话
active_sessions = {}
session_lock = threading.Lock()

# Session 超时时间（秒）- 超过此时间没有活动的 session 会被自动清理
SESSION_TIMEOUT = 60

class StartGenerationRequest(BaseModel):
    prompt: str
    num_blocks: int = 25
    num_denoising_steps: int = 4
    strength: float = 0.45
    seed: Optional[int] = None
    start_frame: Optional[str] = None  # base64 encoded

class FrameRequest(BaseModel):
    session_id: str
    image: Optional[str] = None  # base64 encoded
    prompt: Optional[str] = None
    strength: Optional[float] = None

class GenerationSession:
    def __init__(self, session_id: str, model_instance):
        self.session_id = session_id
        self.model = model_instance
        self.initialized = False
        self.current_block = 0
        self.num_blocks = 25
        self.pending_frames = []  # 待发送的帧
        self.lock = threading.Lock()
        self.last_activity = time.time()  # 最后活动时间
    
    def touch(self):
        """更新最后活动时间"""
        self.last_activity = time.time()
    
    def is_expired(self):
        """检查 session 是否超时"""
        return time.time() - self.last_activity > SESSION_TIMEOUT

@app.post("/api/generate/start")
async def api_start_generation(req: StartGenerationRequest):
    """开始生成会话"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session_id = str(uuid.uuid4())[:8]
    
    # 创建会话
    session = GenerationSession(session_id, model)
    session.num_blocks = req.num_blocks
    
    # 处理起始帧
    start_frame = None
    if req.start_frame:
        start_frame_bytes = base64.b64decode(req.start_frame)
        start_frame = model.process_frame_bytes(start_frame_bytes)
    
    # 初始化生成
    await asyncio.to_thread(
        model.initialize_generation,
        prompt=req.prompt,
        start_frame=start_frame,
        num_inference_steps=req.num_denoising_steps,
        strength=req.strength,
        seed=req.seed
    )
    
    session.initialized = True
    
    # 生成第一个 block
    frames = await asyncio.to_thread(
        model.generate_next_block,
        input_frame=None
    )
    
    # 转换帧为 base64
    for frame in frames:
        frame_bytes = model.frame_to_bytes(frame)
        session.pending_frames.append(base64.b64encode(frame_bytes).decode())
    
    session.current_block = 1
    
    with session_lock:
        active_sessions[session_id] = session
    
    print(f"[HTTP] Session {session_id} started, generated block 0")
    
    return {
        "session_id": session_id,
        "status": "started",
        "frames_ready": len(session.pending_frames)
    }

@app.post("/api/generate/frame")
async def api_generate_frame(req: FrameRequest):
    """发送帧并获取生成的帧"""
    with session_lock:
        session = active_sessions.get(req.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 更新活动时间
    session.touch()
    
    # 更新参数
    if req.prompt:
        session.model.prompt = req.prompt
    if req.strength:
        session.model.strength = req.strength
    
    # 如果有输入帧，生成下一个 block
    if req.image and session.current_block < session.num_blocks:
        input_frame_bytes = base64.b64decode(req.image)
        input_frame = session.model.process_frame_bytes(input_frame_bytes)
        
        frames = await asyncio.to_thread(
            session.model.generate_next_block,
            input_frame=input_frame
        )
        
        with session.lock:
            for frame in frames:
                frame_bytes = session.model.frame_to_bytes(frame)
                session.pending_frames.append(base64.b64encode(frame_bytes).decode())
        
        session.current_block += 1
    
    # 返回待发送的帧
    with session.lock:
        frames_to_send = session.pending_frames[:5]  # 每次最多返回5帧
        session.pending_frames = session.pending_frames[5:]
    
    return {
        "session_id": req.session_id,
        "current_block": session.current_block,
        "total_blocks": session.num_blocks,
        "frames": frames_to_send,
        "complete": session.current_block >= session.num_blocks and len(session.pending_frames) == 0
    }

@app.get("/api/generate/frames/{session_id}")
async def api_get_frames(session_id: str, count: int = 5):
    """获取生成的帧（不发送新输入）"""
    with session_lock:
        session = active_sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 更新活动时间
    session.touch()
    
    with session.lock:
        frames_to_send = session.pending_frames[:count]
        session.pending_frames = session.pending_frames[count:]
    
    return {
        "session_id": session_id,
        "frames": frames_to_send,
        "frames_remaining": len(session.pending_frames),
        "complete": session.current_block >= session.num_blocks and len(session.pending_frames) == 0
    }

@app.post("/api/generate/stop/{session_id}")
async def api_stop_generation(session_id: str):
    """停止生成会话"""
    with session_lock:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            # 清理推理显存
            if hasattr(session.model, 'cleanup_inference'):
                session.model.cleanup_inference()
            del active_sessions[session_id]
            print(f"[HTTP] Session {session_id} stopped, memory cleaned")
            return {"status": "stopped"}
    
    return {"status": "not_found"}


@app.websocket("/ws/video-gen")
async def websocket_video_gen(websocket: WebSocket):
    """WebSocket 处理 - 使用本地 GPU"""
    await websocket.accept()
    
    active_websockets.add(websocket)
    print(f"WebSocket connected. Active connections: {len(active_websockets)}")
    
    try:
        # 检查模型是否已加载
        if model is None:
            await websocket.close(code=1011, reason="Model not loaded yet, please wait...")
            return
        
        inference_model = model
        
        # 发送 ready 信号
        await websocket.send_text(json.dumps({"status": "ready"}))
        print("Sent ready signal to client")
        
        # 初始化标志
        initialized = False
        prompt = ""
        num_blocks = 25
        current_block = 0
        
        while True:
            # 接收消息
            data = await websocket.receive_bytes()
            
            # 解析 msgpack
            message = msgpack.unpackb(data, raw=False)
            
            # 初始化参数
            if not initialized and "prompt" in message:
                prompt = message.get("prompt", "")
                num_blocks = message.get("num_blocks", 25)
                num_inference_steps = message.get("num_denoising_steps", 4)
                strength = message.get("strength", 0.45)
                seed = message.get("seed")
                start_frame = message.get("start_frame")  # 可能是 bytes
                
                print(f"Initializing: prompt='{prompt}', num_blocks={num_blocks}")
                
                # 初始化生成
                await asyncio.to_thread(
                    inference_model.initialize_generation,
                    prompt=prompt,
                    start_frame=start_frame,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    seed=seed
                )
                
                initialized = True
                current_block = 0
                
                # 立即生成第一个 block
                print(f"Generating block 0/{num_blocks}")
                frames = await asyncio.to_thread(
                    inference_model.generate_next_block,
                    input_frame=None
                )
                
                # 发送帧
                for frame in frames:
                    frame_bytes = inference_model.frame_to_bytes(frame)
                    await websocket.send_bytes(frame_bytes)
                
                current_block += 1
                
            # 更新参数（prompt 或 num_blocks 变化）
            elif initialized and "prompt" in message and "image" not in message:
                new_prompt = message.get("prompt")
                new_num_blocks = message.get("num_blocks")
                
                if new_prompt != prompt:
                    print(f"Prompt updated: '{new_prompt}'")
                    prompt = new_prompt
                    inference_model.prompt = prompt
                    
                if new_num_blocks != num_blocks:
                    print(f"num_blocks updated: {new_num_blocks}")
                    num_blocks = new_num_blocks
            
            # 接收输入帧（video-to-video 或 webcam 模式）
            elif initialized and "image" in message:
                input_frame_bytes = message["image"]
                strength = message.get("strength", 0.45)
                
                # 更新 strength
                inference_model.strength = strength
                
                # 可能还有 prompt 更新
                if "prompt" in message:
                    inference_model.prompt = message["prompt"]
                if "num_blocks" in message:
                    num_blocks = message["num_blocks"]
                
                # 生成下一个 block
                if current_block < num_blocks:
                    input_frame = inference_model.process_frame_bytes(input_frame_bytes)
                    
                    frames = await asyncio.to_thread(
                        inference_model.generate_next_block,
                        input_frame=input_frame
                    )
                    
                    # 发送帧
                    for frame in frames:
                        frame_bytes = inference_model.frame_to_bytes(frame)
                        await websocket.send_bytes(frame_bytes)
                    
                    current_block += 1
                    
                    if current_block >= num_blocks:
                        print(f"Generation complete: {current_block} blocks")
                        
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        active_websockets.discard(websocket)
        # 清理推理显存
        if model is not None and hasattr(model, 'cleanup_inference'):
            model.cleanup_inference()
            print("Inference memory cleaned")
        print(f"WebSocket disconnected. Active connections: {len(active_websockets)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
