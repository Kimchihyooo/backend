import requests

# Your live Modal GPU server URL!
MODAL_GPU_URL = "https://kimchihyooo--deepfake-detector-gpu-worker-deepfakescanne-2732a9.modal.run"

def run_deepfake_analysis(video_url, is_verified_source=False, **kwargs):
    """
    Sends the video URL to the Modal GPU for heavy AI processing.
    (We include **kwargs just in case your main app still tries to pass 
    old arguments like yolo_model or deepfake_model).
    """
    print(f"--- Sending Video to GPU for Analysis: {video_url} ---")
    
    payload = {
        "video_url": video_url,
        "is_verified_source": is_verified_source
    }
    
    try:
        # High timeout (600s = 10 minutes) to give the GPU time to download and scan
        response = requests.post(MODAL_GPU_URL, json=payload, timeout=600)
        response.raise_for_status() 
        return response.json()
        
    except Exception as e:
        print(f"GPU Server Error: {e}")
        return {
            "verdict": "Error",
            "confidence": 0,
            "interpretation": f"Failed to process video on GPU: {str(e)}",
            "suspicious_frames": [],
            "timeline_graph": None
        }