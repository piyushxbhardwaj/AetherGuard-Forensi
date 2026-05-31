import sys
import os

# Dynamically add project root to path for stable imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import torch
import time
import requests
import io
import base64
import traceback
from PIL import Image
from models.model import load_hf_model
from utils.inference import InferenceEngine

app = FastAPI(title="Deepfake Detection API (Hugging Face Transformers)")
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "AetherGuard API running",
        "docs": "/docs",
        "health": "/api/health"
    }

# Global Exception Handlers
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Unhandled Exception: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Server-side crash", "detail": str(exc), "traceback": traceback.format_exc()}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Server initializing on device: {device}")

# 2. ASYNC MODEL LOADING (AVOID DEPLOYMENT TIMEOUTS)
inference_engine = None

@app.on_event("startup")
async def startup_event():
    import threading
    def load_engine():
        global inference_engine
        try:
            print("⏳ Background: Loading AI Engine...")
            model, processor = load_hf_model(device=device)
            inference_engine = InferenceEngine(model=model, processor=processor, device=device)
            print("🚀 AI Engine fully loaded and stabilized.")
        except Exception as e:
            print(f"🛑 CRITICAL AI MODEL FAILURE: {str(e)}")
            traceback.print_exc()
            # Mock fallback
            class MockInferenceEngine:
                def predict_image(self, *args, **kwargs):
                    return {"label": "ERROR: MODEL OFFLINE", "confidence": 0, "error": str(e)}
                def predict_video(self, *args, **kwargs):
                    return {"label": "ERROR: MODEL OFFLINE", "confidence": 0, "error": str(e)}
            inference_engine = MockInferenceEngine()
    
    # Run in background thread to not block the main event loop
    threading.Thread(target=load_engine).start()

@app.get("/api/health")
def health_check():
    return {
        "status": "ok", 
        "device": device, 
        "engine": "Transformers/ViT"
    }

# 3. ROBUST PREDICT IMAGE ENDPOINT
@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Invalid file type. Please upload an image."}
        
        file_path = os.path.join(UPLOAD_DIR, f"img_{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if inference_engine is None:
            return {"error": "AI Engine is still initializing. Please wait a moment."}
        
        results = inference_engine.predict_image(file_path, threshold=threshold)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return results
    except Exception as e:
        print(f"Endpoint Error (Image): {str(e)}")
        return {"error": str(e)}

# 4. ROBUST PREDICT VIDEO ENDPOINT
@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        if not file.content_type.startswith("video/"):
            return {"error": "Invalid file type. Please upload a video."}
        
        file_path = os.path.join(UPLOAD_DIR, f"vid_{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if inference_engine is None:
            return {"error": "AI Engine is still initializing. Please wait a moment."}
            
        # Process video with 15-frame sampling
        results = inference_engine.predict_video(file_path, sample_rate=15, threshold=threshold)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return results
    except Exception as e:
        print(f"Endpoint Error (Video): {str(e)}")
        return {"error": str(e)}

# 5. STABLE AI FACE TEST ENDPOINT (DASHBOARD TEST)
@app.get("/api/test/ai-face")
async def test_ai_face(threshold: float = 0.5):
    """
    🎯 GOAL: Stable, safe, and works without upload.
    Cycles through GAN providers with a local fallback.
    """
    sources = [
        "https://thispersondoesnotexist.com/",
        "https://fakeface.rest/face/view"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"
    }

    last_error = "No image source reachable"
    
    # 5.1 Remote Fetch Logic
    for url in sources:
        try:
            print(f"Requesting AI Face from provider: {url}")
            response = requests.get(url, headers=headers, timeout=12, verify=True)
            if not response.ok:
                response = requests.get(url, headers=headers, timeout=12, verify=False)
            
            response.raise_for_status()
            
            img_bytes = io.BytesIO(response.content)
            img = Image.open(img_bytes).convert("RGB")
            
            if inference_engine is None:
                return {"error": "AI Engine still initializing..."}
                
            # Predict WITHOUT fallback to ensure face-specific result
            results = inference_engine.predict_image(img, threshold=threshold, allow_fallback=False)
            
            # Return final result with base64 encoded image for UI
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                **results,
                "image_base64": f"data:image/jpeg;base64,{img_base64}"
            }
        except Exception as e:
            print(f"Provider {url} failed: {e}")
            last_error = str(e)
            continue

    # 5.2 LOCAL FALLBACK (REQ 2 & 3)
    local_path = "test_fake.jpg" 
    if os.path.exists(local_path):
        try:
            print(f"Using local target fallback: {local_path}")
            if inference_engine is None:
                return {"error": "AI Engine still initializing..."}
                
            img = Image.open(local_path).convert("RGB")
            results = inference_engine.predict_image(img, threshold=threshold, allow_fallback=False)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                **results,
                "image_base64": f"data:image/jpeg;base64,{img_base64}",
                "source": "local_fallback"
            }
        except Exception as e:
            last_error += f" | Local error: {str(e)}"

    # 5.3 FINAL ERROR RESPONSE (REQ 1)
    return {
        "label": "Source Error",
        "confidence": 0.0,
        "error": f"Failed to fetch test target: {last_error}"
    }

# 6. VIDEO PROXY ENDPOINT
@app.get("/api/proxy/video")
async def proxy_video(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, timeout=20, headers=headers)
        response.raise_for_status()
        return StreamingResponse(
            response.iter_content(chunk_size=65536), 
            media_type=response.headers.get("content-type", "video/mp4")
        )
    except Exception as e:
        print(f"Proxy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable (Render/Vercel) or fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting AetherGuard Inference Engine on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
