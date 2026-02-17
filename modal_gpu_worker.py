# modal_gpu_worker.py

import os
import io
import json
import base64
import tempfile
import asyncio
from collections import deque

# Modal and FastAPI
import modal
from fastapi import Request

# --- 1. MODAL APP CONFIGURATION ---
app = modal.App(name="deepfake-detector-gpu-worker")

# Define the container environment. 
# Added libglib2.0-0 and libgl1 to permanently fix the libxcb.so.1 OpenCV error!
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libgl1", "libglib2.0-0", "procps") 
    .run_commands("pip install tensorflow[and-cuda]==2.15.0 --extra-index-url https://pypi.nvidia.com")
    .pip_install(
        "fastapi[standard]", # <--- ADD THIS LINE RIGHT HERE
        "ultralytics==8.0.222",
        "yt-dlp==2024.3.10",
        "opencv-python-headless==4.8.1.78", 
        "numpy==1.26.4",
        "matplotlib==3.8.4",
        "Pillow==10.3.0",
        "scikit-learn==1.4.2",
        "h5py==3.10.0"
    )
    .add_local_dir("backend/Models/Deepfake", remote_path="/models")
)

# --- 2. CONFIGURATION THRESHOLDS ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
FRAME_INTERVAL = 5        
SUSPICIOUS_THRESHOLD = 0.85 
MAX_EVIDENCE_FRAMES = 8  
DETECTION_SIZE = 1080     
REAL_STREAK_LIMIT = 30    
WINDOW_SIZE = 45          
TRIGGER_THRESHOLD = 35    
MIN_DEEPFAKE_FRAMES = 45  

# --- 3. GRAPH HELPER FUNCTION ---
def generate_timeline_graph(scores):
    import matplotlib.pyplot as plt
    if not scores: return None
    
    plt.figure(figsize=(10, 1.5))
    colors = []
    for score in scores:
        if score > 0.85: colors.append('#ef4444') 
        elif score > 0.60: colors.append('#f59e0b') 
        else: colors.append('#22c55e') 
        
    plt.bar(range(len(scores)), [1]*len(scores), color=colors, width=1.0)
    plt.xlim(0, len(scores))
    plt.yticks([]) 
    plt.xlabel("Video Frames")
    plt.title("Deepfake Detection Timeline", fontsize=10)
    plt.box(False) 
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- 4. THE GPU WORKER CLASS ---
# This class runs entirely on Modal's NVIDIA GPUs
@app.cls(
    gpu="T4", 
    image=image, 
    timeout=1200, 
)
class DeepfakeScanner:
    
    @modal.enter()
    def load_models(self):
        """Runs once when the container boots. Loads YOLO & MobileNetV2 into GPU memory."""
        print("Loading AI Models into GPU memory...")
        import tensorflow as tf
        from ultralytics import YOLO
        import numpy as np
        
        # Load models from the mounted folder
        self.yolo_model = YOLO("/models/Deepfake/model.pt")
        
        # Ensure the filename matches what is inside your Models folder!
        self.deepfake_model = tf.keras.models.load_model("/models/Deepfake/mobilenetv2_best_model.keras") 
        
        # Warm up models to avoid "cold start" lag on the first scan
        self.yolo_model(np.zeros((640, 640, 3)), verbose=False)
        self.deepfake_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
        print("Models successfully loaded and warmed up!")

    @modal.web_endpoint(method="POST")
    async def analyze_video(self, request: Request):
        """The API Endpoint that Railway will call."""
        import yt_dlp
        import cv2
        import numpy as np
        from keras.applications.mobilenet_v2 import preprocess_input
        from keras.utils import img_to_array
        
        payload = await request.json()
        video_url = payload.get("video_url")
        is_verified_source = payload.get("is_verified_source", False)

        if not video_url:
            return {"error": "video_url is missing"}, 400

        # 1. Download Video securely via yt-dlp
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'noplaylist': True, 
            'quiet': True,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'video.mp4')
            ydl_opts['outtmpl'] = video_path
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as e:
                return {"error": f"Failed to download video: {str(e)}"}, 500
                
            # 2. Start Video Analysis
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}, 500

            all_scores = []
            suspicious_frames_base64 = []
            frame_count = 0
            fake_frames_count = 0
            noise_frames = []
            real_streak = 0
            stopped_at_frame = 0
            frame_history = deque(maxlen=WINDOW_SIZE)
            hard_deepfake_verdict = False
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    if frame_count % FRAME_INTERVAL != 0: continue

                    h, w = frame.shape[:2]
                    scale = 1.0
                    detect_frame = frame

                    if w > DETECTION_SIZE:
                        scale = DETECTION_SIZE / w
                        new_h = int(h * scale)
                        detect_frame = cv2.resize(frame, (DETECTION_SIZE, new_h))

                    # Face Detection
                    results = self.yolo_model(detect_frame, verbose=False)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        box = results[0].boxes[0]
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().astype(int)
                        
                        if scale != 1.0:
                            x1, y1 = int(x1 / scale), int(y1 / scale)
                            x2, y2 = int(x2 / scale), int(y2 / scale)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0: continue

                        # Deepfake Classification
                        face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
                        img_array = img_to_array(face_resized)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = preprocess_input(img_array)

                        pred = self.deepfake_model.predict(img_array, verbose=0)[0][0]
                        all_scores.append(float(pred)) 

                        if pred > SUSPICIOUS_THRESHOLD:
                            frame_history.append(1) 
                            fake_frames_count += 1
                            real_streak = 0
                            
                            if len(suspicious_frames_base64) < MAX_EVIDENCE_FRAMES:
                                evidence_frame = frame.copy()
                                cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                label = f"Check: {pred*100:.0f}%"
                                cv2.putText(evidence_frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                _, buffer = cv2.imencode('.jpg', evidence_frame)
                                suspicious_frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
                        elif pred > 0.60:
                            frame_history.append(0) 
                            real_streak = 0
                            noise_frames.append(frame_count)
                        else:
                            frame_history.append(0) 
                            real_streak += 1

                        # Early Stopping Rules
                        if len(frame_history) == WINDOW_SIZE and sum(frame_history) >= TRIGGER_THRESHOLD:
                            hard_deepfake_verdict = True
                            stopped_at_frame = frame_count
                            break 

                        if real_streak >= REAL_STREAK_LIMIT:
                            stopped_at_frame = frame_count
                            break 

            finally:
                cap.release()

            # 3. Compile Final Results
            if not all_scores:
                return {
                    "verdict": "Uncertain",
                    "confidence": 0,
                    "interpretation": "No faces detected in the analyzed frames.",
                    "suspicious_frames": [],
                    "timeline_graph": None
                }

            graph_base64 = generate_timeline_graph(all_scores)
            
            if hard_deepfake_verdict or fake_frames_count >= MIN_DEEPFAKE_FRAMES:
                verdict = "Visual Anomalies Detected (Source Verified)" if is_verified_source else "Likely Deepfake"
                confidence = 99.0 if hard_deepfake_verdict else (max(all_scores) * 100)
                interpretation = "High probability of manipulation detected. A sustained sequence of deepfake artifacts was found."
            elif fake_frames_count > 0:
                if is_verified_source:
                    verdict = "Likely Real (Noise Detected)"
                    confidence = max(0, 100 - (fake_frames_count * 2)) 
                    interpretation = f"Source is verified. Detected {fake_frames_count} frames of minor artifacts, likely standard video compression."
                else:
                    verdict = "Suspicious"
                    confidence = max(0, 100 - (fake_frames_count * 3))
                    interpretation = f"Detected {fake_frames_count} anomalous frames. Most likely a deepfake, and source is unverified."
            else:
                verdict = "Likely Real"
                confidence = 100.0
                interpretation = "Overall, the video appears authentic with no signs of Deepfake manipulation."

            return {
                "verdict": verdict,
                "confidence": round(confidence, 1),
                "interpretation": interpretation,
                "suspicious_frames": suspicious_frames_base64,
                "noise_frames": noise_frames,
                "frame_count": fake_frames_count,
                "timeline_graph": graph_base64
            }