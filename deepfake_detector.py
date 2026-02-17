import cv2
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import tensorflow as tf
from ultralytics import YOLO
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array

# --- 1. CONFIGURATION ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
FRAME_INTERVAL = 5        
SUSPICIOUS_THRESHOLD = 0.85 
MAX_EVIDENCE_FRAMES = 8  
DETECTION_SIZE = 1080     
REAL_STREAK_LIMIT = 30    

# UPDATED THRESHOLDS
WINDOW_SIZE = 45          # Increased window size to 1 second of frames (30fps)
TRIGGER_THRESHOLD = 35    # Need 20/30 frames in a window to early-stop
MIN_DEEPFAKE_FRAMES = 45  # Need at least 45 total anomalous frames (1.5s) to trigger Deepfake

# --- 2. GRAPH HELPER FUNCTION ---
def generate_timeline_graph(scores, frame_interval):
    if not scores: return None
    
    plt.figure(figsize=(10, 1.5))
    
    colors = []
    for score in scores:
        if score > 0.85: colors.append('#ef4444') # Red (Deepfake)
        elif score > 0.60: colors.append('#f59e0b') # Yellow (Noise)
        else: colors.append('#22c55e') # Green (Clean)
        
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

# --- 3. MAIN ANALYSIS LOOP ---
# NOTE: Added 'is_verified_source=False' to handle Source Verification logic
def run_deepfake_analysis(video_path, yolo_model, deepfake_model, is_verified_source=False):
    print(f"--- Scanning Video: {video_path} ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

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

            results = yolo_model(detect_frame, verbose=False)
            
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

                face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
                img_array = img_to_array(face_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                pred = deepfake_model.predict(img_array, verbose=0)[0][0]
                all_scores.append(pred)

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

                # NEW EARLY STOP LOGIC: Only stop if we see a prolonged deepfake streak
                if len(frame_history) == WINDOW_SIZE and sum(frame_history) >= TRIGGER_THRESHOLD:
                    hard_deepfake_verdict = True
                    stopped_at_frame = frame_count
                    print(f"--- 🔴 DEEPFAKE EARLY STOP at Frame {stopped_at_frame} ---")
                    break 

                if real_streak >= REAL_STREAK_LIMIT:
                    stopped_at_frame = frame_count
                    print(f"--- ✅ REAL VIDEO EARLY STOP at Frame {stopped_at_frame} ---")
                    break 

    finally:
        cap.release()

    if not all_scores:
        return {
            "verdict": "Uncertain",
            "confidence": 0,
            "interpretation": "No faces detected.",
            "suspicious_frames": [],
            "timeline_graph": None
        }

    # Ensure the helper function is called correctly
    graph_base64 = generate_timeline_graph(all_scores, FRAME_INTERVAL)
    
    # --- UPDATED FINAL VERDICT LOGIC (DYNAMIC THRESHOLD) ---
    if hard_deepfake_verdict or fake_frames_count >= MIN_DEEPFAKE_FRAMES:
        verdict = "Likely Deepfake" if not is_verified_source else "Visual Anomalies Detected (Source Verified)"
        confidence = 99.0 if hard_deepfake_verdict else (max(all_scores) * 100)
        interpretation = "High probability of manipulation detected. A sustained sequence of deepfake artifacts was found."
    
    elif fake_frames_count > 0:
        # Micro-anomaly detected (Less than 45 frames)
        if is_verified_source:
            verdict = "Likely Real (Noise Detected)"
            confidence = max(0, 100 - (fake_frames_count * 2)) 
            interpretation = f"Source is verified. Detected {fake_frames_count} frames of minor artifacts, likely standard video compression or motion blur."
        else:
            verdict = "Suspicious"
            confidence = max(0, 100 - (fake_frames_count * 3))
            interpretation = f"Detected {fake_frames_count} anomalous frames. most likely a deepfake, and source is unverified."
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