import os
import sqlite3
import base64
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, Cookie, HTTPException, WebSocket, Header
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
SPACE_HOST = os.getenv("SPACE_HOST", "localhost:7860")
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
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_pro BOOLEAN NOT NULL,
            generation_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_username_date 
        ON generations(username, generation_date)
    """)
    conn.commit()
    conn.close()

init_db()

def get_daily_usage(username: str, date: str = None) -> int:
    """Get number of generations used today by user"""
    if date is None:
        date = datetime.now().date().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM generations WHERE username = ? AND generation_date = ?",
        (username, date)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count

def record_generation(username: str, is_pro: bool):
    """Record a new generation - called every time user clicks 'Start Generation'"""
    date = datetime.now().date().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO generations (username, is_pro, generation_date) VALUES (?, ?, ?)",
        (username, is_pro, date)
    )
    conn.commit()
    conn.close()

def can_start_generation(username: str, is_pro: bool) -> tuple[bool, int, int]:
    """Check if user can start a new generation. Returns (can_start, used, limit)"""
    used = get_daily_usage(username)
    limit = 15 if is_pro else 1
    return used < limit, used, limit

def get_origin_from_request(request: Request) -> str:
    """Get the origin (scheme + host) from the request, detecting HTTPS from proxy headers"""
    # Check proxy headers for original protocol (common when behind reverse proxy)
    proto = request.headers.get("X-Forwarded-Proto", "")
    ssl = request.headers.get("X-Forwarded-Ssl", "")

    # Get host from headers (handles both direct access and proxy)
    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host") or ""

    # Determine scheme
    if proto == "https" or ssl == "on":
        scheme = "https"
    elif ".hf.space" in host or "huggingface.co" in host:
        # Force HTTPS for Hugging Face domains (they always serve over HTTPS)
        scheme = "https"
    else:
        scheme = request.url.scheme or "https"

    # Build origin URL
    if host:
        return f"{scheme}://{host}"

    # Fallback to SPACE_HOST environment variable with HTTPS
    return f"https://{SPACE_HOST}"

def get_token_from_request(cookie_token: Optional[str], auth_header: Optional[str]) -> Optional[str]:
    """Extract access token from either cookie or Authorization header"""
    # Try Authorization header first (Bearer token)
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    # Fall back to cookie
    return cookie_token

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
async def home(request: Request):
    """Home page - client-side auth with popup OAuth"""
    # Dynamically detect origin from request
    origin = get_origin_from_request(request)
    redirect_uri = f"{origin}/oauth/callback"

    # Return template - authentication will be handled client-side
    return templates.TemplateResponse("index.html", {
        "request": request,
        "oauth_client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri
    })

@app.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(request: Request, code: str, state: Optional[str] = None):
    """Handle OAuth callback - returns HTML that posts message to opener window"""
    origin = get_origin_from_request(request)
    redirect_uri = f"{origin}/oauth/callback"

    if not code:
        # Return error HTML that closes popup
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <body>
        <script>
            (function() {{
                console.log('OAuth callback - missing code error');
                const target = window.opener || window.parent || window;
                if (target) {{
                    target.postMessage({{
                        type: 'HF_OAUTH_ERROR',
                        payload: {{ message: 'Missing authorization code' }}
                    }}, '*');
                }}
                setTimeout(function() {{ window.close(); }}, 300);
            }})();
        </script>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)

    try:
        token_data = await exchange_code_for_token(code, redirect_uri)
        access_token = token_data.get("access_token")

        if not access_token:
            raise Exception("No access token received")

        # Get user info
        user_info = await get_user_info(access_token)

        # Return success HTML that posts message to opener
        success_html = f"""
        <!DOCTYPE html>
        <html>
        <body>
        <script>
            (function() {{
                console.log('OAuth callback - sending success message');
                const target = window.opener || window.parent || window;
                if (target) {{
                    console.log('Posting message to target window');
                    target.postMessage({{
                        type: 'HF_OAUTH_SUCCESS',
                        payload: {{
                            token: {json.dumps(access_token)},
                            username: {json.dumps(user_info["username"])},
                            is_pro: {json.dumps(user_info["is_pro"])},
                            fullname: {json.dumps(user_info["fullname"])},
                            avatar: {json.dumps(user_info.get("avatar"))}
                        }}
                    }}, '*');
                }}
                setTimeout(function() {{
                    console.log('Closing popup window');
                    window.close();
                }}, 300);
            }})();
        </script>
        </body>
        </html>
        """
        return HTMLResponse(content=success_html)

    except Exception as e:
        print(f"OAuth callback error: {e}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <body>
        <script>
            (function() {{
                console.log('OAuth callback - error:', {json.dumps(str(e))});
                const target = window.opener || window.parent || window;
                if (target) {{
                    target.postMessage({{
                        type: 'HF_OAUTH_ERROR',
                        payload: {{ message: {json.dumps(str(e))} }}
                    }}, '*');
                }}
                setTimeout(function() {{ window.close(); }}, 300);
            }})();
        </script>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)

@app.get("/api/whoami")
async def whoami_endpoint(authorization: Optional[str] = Header(None)):
    """Validate token and return user info"""
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")

    # Extract Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    access_token = parts[1]

    try:
        user_info = await get_user_info(access_token)
        # Get usage info
        can_start, used, limit = can_start_generation(user_info["username"], user_info["is_pro"])

        return {
            "username": user_info["username"],
            "fullname": user_info["fullname"],
            "is_pro": user_info["is_pro"],
            "avatar": user_info.get("avatar"),
            "can_start": can_start,
            "sessions_used": used,
            "sessions_limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@app.post("/api/start-session")
async def start_session(access_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """Start a new generation - counts towards daily limit"""
    token = get_token_from_request(access_token, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_info = await get_user_info(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    can_start, used, limit = can_start_generation(user_info["username"], user_info["is_pro"])
    
    if not can_start:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached. You've used {used}/{limit} generations today."
        )
    
    # Record this generation
    record_generation(user_info["username"], user_info["is_pro"])
    
    # Get updated count
    new_count = get_daily_usage(user_info["username"])
    
    return {
        "success": True,
        "sessions_used": new_count,
        "sessions_limit": limit
    }

@app.get("/api/check-limits")
async def check_limits(access_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """Check current usage limits"""
    token = get_token_from_request(access_token, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_info = await get_user_info(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    can_start, used, limit = can_start_generation(user_info["username"], user_info["is_pro"])
    
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
    can_start, used, limit = can_start_generation(user_info["username"], user_info["is_pro"])
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