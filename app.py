import os
import sqlite3
import base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, Cookie, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import whoami
import httpx
import websockets

app = FastAPI()
templates = Jinja2Templates(directory=".")

# OAuth configuration from HF Spaces environment
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.getenv("OAUTH_SCOPES", "openid profile")
SPACE_HOST = os.getenv("SPACE_HOST", "multimodalart-krea-realtime-video.hf.space")
OPENID_PROVIDER_URL = os.getenv("OPENID_PROVIDER_URL", "https://huggingface.co")

# FAL API Key from environment
FAL_API_KEY = os.getenv("FAL_API_KEY", "")

# Database setup
DB_PATH = "/data/usage.db"

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_pro BOOLEAN NOT NULL,
            session_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, session_date)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_daily_usage(username: str, date: str = None) -> int:
    """Get number of sessions used today by user"""
    if date is None:
        date = datetime.now().date().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM sessions WHERE username = ? AND session_date = ?",
        (username, date)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count

def record_session(username: str, is_pro: bool):
    """Record a new session"""
    date = datetime.now().date().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO sessions (username, is_pro, session_date) VALUES (?, ?, ?)",
            (username, is_pro, date)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def can_start_session(username: str, is_pro: bool) -> tuple[bool, int, int]:
    """Check if user can start a new session. Returns (can_start, used, limit)"""
    used = get_daily_usage(username)
    limit = 15 if is_pro else 1
    return used < limit, used, limit

async def exchange_code_for_token(code: str, redirect_uri: str) -> dict:
    """Exchange OAuth code for access token"""
    token_url = f"{OPENID_PROVIDER_URL}/oauth/token"
    
    credentials = f"{OAUTH_CLIENT_ID}:{OAUTH_CLIENT_SECRET}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {b64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": OAUTH_CLIENT_ID
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data, headers=headers)
        response.raise_for_status()
        return response.json()

async def get_user_info(access_token: str) -> dict:
    """Get user info from access token using whoami"""
    try:
        user_data = whoami(token=access_token)
        return {
            "username": user_data.get("name"),
            "is_pro": user_data.get("isPro", False),
            "avatar": user_data.get("avatarUrl"),
            "fullname": user_data.get("fullname", user_data.get("name")),
            "email": user_data.get("email")
        }
    except Exception as e:
        print(f"Failed to get user info: {e}")
        raise HTTPException(status_code=401, detail="Failed to get user information")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, access_token: Optional[str] = Cookie(None)):
    """Home page - check auth and show app or login"""
    
    if not access_token:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "authenticated": False,
            "oauth_client_id": OAUTH_CLIENT_ID,
            "redirect_uri": f"https://{SPACE_HOST}/oauth/callback",
            "space_host": SPACE_HOST
        })
    
    try:
        user_info = await get_user_info(access_token)
    except:
        response = templates.TemplateResponse("index.html", {
            "request": request,
            "authenticated": False,
            "oauth_client_id": OAUTH_CLIENT_ID,
            "redirect_uri": f"https://{SPACE_HOST}/oauth/callback",
            "space_host": SPACE_HOST,
            "error": "Session expired. Please login again."
        })
        response.delete_cookie("access_token")
        return response
    
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "authenticated": True,
        "user": user_info,
        "can_start": can_start,
        "sessions_used": used,
        "sessions_limit": limit
    })

@app.get("/oauth/callback")
async def oauth_callback(code: str, state: Optional[str] = None):
    """Handle OAuth callback from Hugging Face"""
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    
    redirect_uri = f"https://{SPACE_HOST}/oauth/callback"
    
    try:
        token_data = await exchange_code_for_token(code, redirect_uri)
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=30 * 24 * 60 * 60
        )
        
        return response
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.post("/api/start-session")
async def start_session(access_token: Optional[str] = Cookie(None)):
    """Start a new generation session"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_info = await get_user_info(access_token)
    except:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    
    if not can_start:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached. You've used {used}/{limit} sessions today."
        )
    
    record_session(user_info["username"], user_info["is_pro"])
    
    return {
        "success": True,
        "sessions_used": used + 1,
        "sessions_limit": limit
    }

@app.get("/api/check-limits")
async def check_limits(access_token: Optional[str] = Cookie(None)):
    """Check current usage limits"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_info = await get_user_info(access_token)
    except:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    
    return {
        "can_start": can_start,
        "sessions_used": used,
        "sessions_limit": limit,
        "is_pro": user_info["is_pro"]
    }

@app.post("/api/logout")
async def logout():
    """Logout user"""
    response = JSONResponse({"success": True})
    response.delete_cookie("access_token")
    return response

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "oauth_enabled": bool(OAUTH_CLIENT_ID),
        "fal_api_key_configured": bool(FAL_API_KEY)
    }

@app.websocket("/ws/video-gen")
async def websocket_video_gen(websocket: WebSocket):
    """WebSocket proxy to FAL API - keeps API key secret"""
    from fastapi import WebSocket
    import websockets
    import json
    
    await websocket.accept()
    
    # Get user from cookie
    access_token = websocket.cookies.get("access_token")
    if not access_token:
        await websocket.close(code=1008, reason="Not authenticated")
        return
    
    try:
        user_info = await get_user_info(access_token)
    except:
        await websocket.close(code=1008, reason="Invalid session")
        return
    
    # Check if user can start session
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    if not can_start:
        await websocket.close(code=1008, reason=f"Daily limit reached ({used}/{limit})")
        return
    
    if not FAL_API_KEY:
        await websocket.close(code=1011, reason="FAL API key not configured")
        return
    
    # Fetch temporary FAL token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://rest.alpha.fal.ai/tokens/",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Key {FAL_API_KEY}"
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
    
    try:
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
            
            # Run both directions concurrently
            import asyncio
            await asyncio.gather(
                client_to_fal(),
                fal_to_client()
            )
    except Exception as e:
        print(f"WebSocket proxy error: {e}")
        await websocket.close(code=1011, reason=str(e))