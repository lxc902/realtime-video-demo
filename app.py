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
    # Return template - authentication will be handled client-side
    return templates.TemplateResponse("index.html", {
        "request": request,
        "oauth_client_id": OAUTH_CLIENT_ID
    })

@app.get("/api/auth/login")
async def auth_login(request: Request, state: Optional[str] = None):
    """OAuth login - stores state in cookie and redirects to HF OAuth"""
    # Dynamically detect origin from request
    origin = get_origin_from_request(request)
    redirect_uri = f"{origin}/oauth/callback"

    # Generate or use provided state
    oauth_state = state or os.urandom(16).hex()

    # Build OAuth authorize URL
    auth_url = f"https://huggingface.co/oauth/authorize"
    auth_url += f"?response_type=code"
    auth_url += f"&client_id={OAUTH_CLIENT_ID}"
    auth_url += f"&redirect_uri={redirect_uri}"
    auth_url += f"&scope=openid profile"
    auth_url += f"&state={oauth_state}"

    # Create response that redirects to HF OAuth
    response = RedirectResponse(url=auth_url, status_code=302)

    # Store state in cookie for validation in callback
    if not state:  # Only set cookie if state wasn't provided
        response.set_cookie(
            key="hf_oauth_state",
            value=oauth_state,
            httponly=True,
            samesite="lax",
            secure=True,
            max_age=300,  # 5 minutes
            path="/"
        )

    return response

@app.post("/api/auth/exchange")
async def auth_exchange(request: Request, code: str, state: str, hf_oauth_state: Optional[str] = Cookie(None)):
    """Exchange OAuth code for access token - called from callback page"""
    print(f"Exchange request - code: {code[:20] if code and len(code) >= 20 else code}..., state: {state}, cookie_state: {hf_oauth_state}")

    # Validate state from cookie
    if not hf_oauth_state or state != hf_oauth_state:
        print(f"State mismatch! URL state: {state}, Cookie state: {hf_oauth_state}")
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    origin = get_origin_from_request(request)
    redirect_uri = f"{origin}/oauth/callback"

    try:
        token_data = await exchange_code_for_token(code, redirect_uri)
        access_token = token_data.get("access_token")

        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        # Get user info
        user_info = await get_user_info(access_token)

        # Return token and user info
        response = JSONResponse({
            "token": access_token,
            "namespace": user_info["username"]
        })
        response.delete_cookie("hf_oauth_state")
        return response

    except Exception as e:
        print(f"Token exchange error: {e}")
        response = JSONResponse(
            {"error": str(e)},
            status_code=400
        )
        response.delete_cookie("hf_oauth_state")
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

@app.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(request: Request):
    """OAuth callback page - exchanges code for token client-side"""
    callback_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Authenticating...</title></head>
    <body style="font-family: sans-serif; padding: 40px; text-align: center;">
        <h2>Authenticating...</h2>
        <p>Please wait while we complete your login.</p>
        <script>
            (async function() {
                const params = new URLSearchParams(window.location.search);
                const code = params.get('code');
                const state = params.get('state');
                const error = params.get('error');

                if (error) {
                    document.body.innerHTML = '<h2>Authentication failed</h2><p>' + error + '</p>';
                    setTimeout(() => window.location.href = '/', 3000);
                    return;
                }

                if (!code || !state) {
                    document.body.innerHTML = '<h2>Authentication failed</h2><p>Missing authorization code</p>';
                    setTimeout(() => window.location.href = '/', 3000);
                    return;
                }

                try {
                    console.log('Exchanging code for token...');
                    // Exchange code for token
                    const response = await fetch('/api/auth/exchange?code=' + code + '&state=' + state, {
                        method: 'POST',
                        credentials: 'same-origin'
                    });

                    console.log('Exchange response status:', response.status);

                    if (!response.ok) {
                        const data = await response.json().catch(() => ({ detail: 'Unknown error' }));
                        console.error('Exchange failed:', data);
                        throw new Error(data.detail || data.error || 'Failed to exchange code for token');
                    }

                    const data = await response.json();
                    console.log('Token exchange successful');

                    // Store in localStorage
                    const authState = {
                        token: data.token,
                        user: { username: data.namespace }
                    };
                    localStorage.setItem('HF_AUTH_STATE', JSON.stringify(authState));
                    console.log('Token saved to localStorage, redirecting to /');

                    // Redirect back to home
                    window.location.href = '/';
                } catch (err) {
                    console.error('OAuth error:', err);
                    document.body.innerHTML = '<h2>Authentication failed</h2><p>' + err.message + '</p><p><a href="/">Return to app</a></p>';
                }
            })();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=callback_html)

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