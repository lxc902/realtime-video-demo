import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import whoami
import secrets

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
        # Already recorded today, that's fine
        pass
    finally:
        conn.close()

def can_start_session(username: str, is_pro: bool) -> tuple[bool, int, int]:
    """Check if user can start a new session. Returns (can_start, used, limit)"""
    used = get_daily_usage(username)
    limit = 15 if is_pro else 1
    return used < limit, used, limit

def verify_token(token: str) -> Optional[dict]:
    """Verify HF token and get user info"""
    try:
        user_info = whoami(token=token)
        return {
            "username": user_info.get("name"),
            "is_pro": user_info.get("isPro", False),
            "avatar": user_info.get("avatarUrl"),
            "fullname": user_info.get("fullname", user_info.get("name"))
        }
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, hf_token: Optional[str] = Cookie(None)):
    """Home page - check auth and show app or login"""
    
    if not hf_token:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "authenticated": False
        })
    
    user_info = verify_token(hf_token)
    if not user_info:
        # Invalid token, clear it
        response = templates.TemplateResponse("index.html", {
            "request": request,
            "authenticated": False,
            "error": "Session expired. Please login again."
        })
        response.delete_cookie("hf_token")
        return response
    
    # Check session limits
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "authenticated": True,
        "user": user_info,
        "can_start": can_start,
        "sessions_used": used,
        "sessions_limit": limit
    })

@app.post("/api/login")
async def login(request: Request):
    """Login with HF token"""
    data = await request.json()
    token = data.get("token", "").strip()
    
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
    
    user_info = verify_token(token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    response = JSONResponse({
        "success": True,
        "user": user_info
    })
    
    # Set secure cookie with token (expires in 30 days)
    response.set_cookie(
        key="hf_token",
        value=token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=30 * 24 * 60 * 60
    )
    
    return response

@app.post("/api/start-session")
async def start_session(hf_token: Optional[str] = Cookie(None)):
    """Start a new generation session"""
    if not hf_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_info = verify_token(hf_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    can_start, used, limit = can_start_session(user_info["username"], user_info["is_pro"])
    
    if not can_start:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached. You've used {used}/{limit} sessions today."
        )
    
    # Record the session
    record_session(user_info["username"], user_info["is_pro"])
    
    return {
        "success": True,
        "sessions_used": used + 1,
        "sessions_limit": limit
    }

@app.get("/api/check-limits")
async def check_limits(hf_token: Optional[str] = Cookie(None)):
    """Check current usage limits"""
    if not hf_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_info = verify_token(hf_token)
    if not user_info:
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
    response.delete_cookie("hf_token")
    return response

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}