# app.py (UPDATED)

import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import uvicorn
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# --- ADD THIS IMPORT ---
from starlette.middleware.sessions import SessionMiddleware
# ---------------------

# Import model loader
from model_loader import get_model_loader

# Import the NEW centralized analysis function
from analysis_helpers import analyze_text_content

# Import the new video analyzer router
from routers import app_video

# Initialize the FastAPI app
app = FastAPI(title="CredibilityScan API")

# --- ADD SESSION MIDDLEWARE ---
app.add_middleware(SessionMiddleware, secret_key="d14cdd028ff805930a8be99c9fc3a161c63f7e41e6e4360db99ba76b03263763")
# ------------------------------

# --- ADD CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
# -------------------------

# ======================== LOAD ALL MODELS ========================
print("Initializing Model Loader...")
model_loader = get_model_loader()
models = model_loader.get_all_models()

# ======================== ENDPOINTS ========================


# ======================== TEXT ANALYSIS ========================

@app.post("/analyze")
async def analyze_news(request: Request, news_text: str = Form(...)):
    
    # 1. Run the analysis (This calls your helper function)
    result_data = await analyze_text_content(news_text, models)
    
    # 2. Add extra data for the "Time Machine" history
    result_data["input_text"] = news_text
    
    # 3. Add a timestamp
    from datetime import datetime
    result_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    
    # 4. Return JSON
    return JSONResponse(content=result_data)

# ======================== VIDEO ANALYSIS ========================

# Check if Deepfake models are loaded instead of Whisper
if models.get("deepfake_model") and models.get("yolo_model"):
    video_router = app_video.create_video_router(**models)
    app.include_router(video_router, tags=["Video Analysis"])
    print("✓ Video analysis router included.")
else:
    # Optional: Include it anyway if you want metadata search to work even without Deepfake models
    # But based on your previous logic, we keep it conditional.
    print("✗ Video analysis router NOT included (Deepfake/YOLO models failed to load).")

# ======================== RUN APP ========================

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)