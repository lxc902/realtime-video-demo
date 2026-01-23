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
    print("   This will take 1-2 minutes on first run")
    print("=" * 60)
    print("")
    model = get_model()
    print("")
    print("=" * 60)
    print("✅ Model loaded successfully!")
    print("🌐 Server is ready to accept connections")
    print("=" * 60)
    print("")

@app.on_event("startup")
async def startup_event():
    """应用启动时的事件"""
    import asyncio
    # 在后台线程加载模型，避免阻塞启动
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_on_startup)


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
        print(f"WebSocket disconnected. Active connections: {len(active_websockets)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
