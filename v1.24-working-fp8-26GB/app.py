import os
from typing import Optional
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import websockets

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
        # Only add CSP to HTML responses
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

# FAL API Key from environment
FAL_API_KEY = os.getenv("FAL_API_KEY", "")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "fal_api_key_configured": bool(FAL_API_KEY)
    }

@app.websocket("/ws/video-gen")
async def websocket_video_gen(websocket: WebSocket, user_fal_key: Optional[str] = None):
    """WebSocket proxy to FAL API - keeps API key secret"""
    from fastapi import WebSocket
    import websockets
    import asyncio

    await websocket.accept()

    # Track this connection
    active_websockets.add(websocket)
    print(f"WebSocket connected. Active connections: {len(active_websockets)}")

    try:
        # Determine which FAL key to use
        if user_fal_key:
            fal_key_to_use = user_fal_key
        else:
            if not FAL_API_KEY:
                await websocket.close(code=1011, reason="FAL API key not configured")
                return
            fal_key_to_use = FAL_API_KEY

        # Fetch temporary FAL token
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://rest.alpha.fal.ai/tokens/",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Key {fal_key_to_use}"
                    },
                    json={
                        "allowed_apps": ["krea-wan-14b"],
                        "token_expiration": 5000
                    }
                )
                response.raise_for_status()
                fal_token = response.json()
        except Exception as e:
            await websocket.close(code=1011, reason=f"Failed to get FAL token: {str(e)}")
            return
    
        # Connect to FAL WebSocket
        fal_ws_url = f"wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={fal_token}"

        async with websockets.connect(fal_ws_url) as fal_ws:
            # Relay messages between client and FAL
            async def client_to_fal():
                try:
                    while True:
                        # Receive from client
                        data = await websocket.receive_bytes()
                        # Forward to FAL
                        await fal_ws.send(data)
                except Exception as e:
                    print(f"Client to FAL error: {e}")
                    raise  # Re-raise to stop both coroutines

            async def fal_to_client():
                try:
                    while True:
                        # Receive from FAL
                        message = await fal_ws.recv()
                        # Forward to client
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except Exception as e:
                    print(f"FAL to client error: {e}")
                    raise  # Re-raise to stop both coroutines

            # Run both directions concurrently - if either fails, both stop
            try:
                await asyncio.gather(
                    client_to_fal(),
                    fal_to_client()
                )
            except Exception:
                # One direction failed, close everything
                pass
    except Exception as e:
        print(f"WebSocket proxy error: {e}")
        await websocket.close(code=1011, reason=str(e))
    finally:
        # Remove from active connections - ALWAYS executes
        active_websockets.discard(websocket)
        print(f"WebSocket disconnected. Active connections: {len(active_websockets)}")