import os
import asyncio
import re
import string
import requests
import json 
from difflib import SequenceMatcher
from urllib.parse import urlparse
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse 
import yt_dlp
from GCustomSearch import fetch_supporting_articles

# Your live Modal GPU Server!
MODAL_GPU_URL = "https://kimchihyooo--deepfake-detector-gpu-worker-deepfakescanne-2732a9.modal.run"

API_KEY = "AIzaSyBXn8fosKRpik2tYEevYvQnMc-np5a2WMM" 
CX_ID = "f48bb9e7a0bf04a69"
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# TRUSTED CHANNEL MAPPING (Shortened here for brevity, keep your full list!)
TRUSTED_CHANNELS = {
    "gma": "GMA News", "abs-cbn": "ABS-CBN News", "abscbn": "ABS-CBN News",
    "rappler": "Rappler", "tv5": "News5", "news5": "News5", "inquirer": "Inquirer.net",
    "philstar": "Philstar", "manila times": "The Manila Times", "cnn philippines": "CNN Philippines",
    "ptv": "PTV Philippines", "anc": "ANC", "dzmm": "DZMM", "super radyo": "Super Radyo dzBB",
    "dzbb": "Super Radyo dzBB", "unang hirit": "Unang Hirit", "24 oras": "24 Oras",
    "saksi": "Saksi", "frontline pilipinas": "Frontline Pilipinas", "bandila": "Bandila",
    "tv patrol": "TV Patrol"
}

TRUSTED_DOMAINS = [
    "gmanetwork.com", "abs-cbn.com", "philstar.com", "inquirer.net", 
    "rappler.com", "verafiles.org", "manilatimes.net", "cnnphilippines.com", 
    "news.tv5.com.ph", "mb.com.ph", "pna.gov.ph", "sunstar.com.ph", 
    "abante.com.ph", "bbc.com", "reuters.com", "dw.com", "aljazeera.com",
    "youtube.com"
]

STOPWORDS = {"akin", "aking", "ako", "alin", "am"} # Keep your full list here!

class QuietLogger:
    def debug(self, msg): pass 
    def warning(self, msg): pass 
    def error(self, msg): print(msg) 

def get_clean_tokens(text):
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    tokens = [w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2]
    return set(tokens)

def scrape_tiktok_oembed(video_url):
    try:
        oembed_url = f"https://www.tiktok.com/oembed?url={video_url}"
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"title": data.get("title", "TikTok Video"), "author": data.get("author_name", "TikTok User"), "thumbnail": data.get("thumbnail_url", ""), "platform": "TikTok", "tags": []}
    except Exception as e:
        print(f"TikTok oEmbed failed: {e}")
    return None

async def analyze_metadata_credibility(video_title, video_tags=[], video_author=""):
    is_trusted_channel = False
    trusted_reason = ""
    clean_author = video_author.lower().strip()
    
    for key, official_name in TRUSTED_CHANNELS.items():
        if key in clean_author:
            is_trusted_channel = True
            trusted_reason = f"Source verified as official {official_name} channel."
            break 

    sentences = re.split(r'(?<=[.!?])\s+', video_title.strip())
    raw_claim = sentences[0] if sentences else ""
    extracted_claim = re.sub(r'^[A-Z\s,]+(?:\s[-—]\s?|\s?[-—]\s)', '', raw_claim).strip()
    if not extracted_claim: extracted_claim = raw_claim

    if not is_trusted_channel:
        return "UNVERIFIED", 20, [], "Source not in trusted database. External search skipped.", extracted_claim

    evidence_list = []
    try:
        results, keywords, extracted_claim = await fetch_supporting_articles(video_title, match_threshold=1.0)
        for item in results:
            link = item.link
            is_trusted_domain = any(trusted in link.lower() for trusted in TRUSTED_DOMAINS)
            status = "VERIFIED" if is_trusted_domain else "RELATED"
            if any(x in item.title.lower() for x in ["false", "hoax", "fake", "fact check", "misleading", "hindi totoo"]):
                status = "DEBUNKED"
            evidence_list.append({"title": item.title, "link": link, "website": urlparse(link).netloc.replace("www.", ""), "displayLink": item.displayLink, "status": status, "snippet": item.snippet, "relevance_score": item.relevance_score, "matched_keywords": item.matched_keywords})
    except Exception as e:
        print(f"GCustomSearch Error: {e}")

    if any(e['status'] == "DEBUNKED" for e in evidence_list): return "DEBUNKED", 95, evidence_list, "Flagged by fact-checkers as false.", extracted_claim
    if is_trusted_channel: return "VERIFIED", 90, evidence_list, trusted_reason, extracted_claim
    if any(e['status'] == "VERIFIED" for e in evidence_list): return "VERIFIED", 85, evidence_list, "Confirmed by trusted news sources.", extracted_claim
    if evidence_list: return "CORROBORATED", 60, evidence_list, "Similar content found online.", extracted_claim
    return "UNVERIFIED", 20, [], "No matching reports found from trusted sources.", extracted_claim

# Notice we removed kwargs from this function
def create_video_router(): 
    router = APIRouter()

    @router.post("/fetch-video-metadata")
    async def fetch_video_metadata(video_url: str = Form(...)):
        ydl_opts = {'quiet': True, 'noplaylist': True, 'skip_download': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return JSONResponse({"status": "success", "title": info.get('title', 'Unknown Title'), "thumbnail": info.get('thumbnail', ''), "duration": info.get('duration', 0), "author": info.get('uploader') or info.get('channel') or "Unknown Source", "platform": info.get('extractor_key', 'Unknown Platform'), "tags": info.get('tags', [])})
        except Exception as e:
            if "tiktok.com" in video_url:
                fb = scrape_tiktok_oembed(video_url)
                if fb: return JSONResponse({"status": "success", **fb, "duration": 0})
            return JSONResponse({"status": "error", "message": f"Could not fetch metadata: {str(e)}"}, status_code=400)

    @router.post("/analyze-video")
    async def analyze_video(request: Request, video_url: str = Form(...)):
        try:
            video_title = "Unknown"; video_tags = []; video_author = "Unknown"
            try:
                with yt_dlp.YoutubeDL({'quiet':True, 'skip_download':True}) as ydl:
                    meta = ydl.extract_info(video_url, download=False)
                    video_title = meta.get('title', 'Unknown')
                    video_tags = meta.get('tags', [])
                    video_author = meta.get('uploader') or meta.get('channel') or "Unknown"
            except:
                if "tiktok.com" in video_url:
                    fb = scrape_tiktok_oembed(video_url)
                    if fb: video_title = fb['title']; video_author = fb['author']

            search_verdict, metadata_confidence, evidence_list, search_reason, extracted_claim = await analyze_metadata_credibility(video_title, video_tags, video_author)
            is_verified = (search_verdict == "VERIFIED")

            print(f"Sending video to GPU for deepfake detection: {video_url}", flush=True)
            
            # FAST AND SIMPLE GPU CALL (No Modal SDK needed!)
            response = requests.post(MODAL_GPU_URL, json={"video_url": video_url, "is_verified_source": is_verified}, timeout=600)
            response.raise_for_status() 
            visual_result = response.json()

            if visual_result.get("error"):
                 raise Exception(f"Modal GPU error: {visual_result['error']}")

            visual_label = visual_result.get("verdict", "Uncertain")
            visual_conf = visual_result.get("confidence", 0)

            final_label = "Uncertain" 
            final_interpretation = []
            colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"} 

            if search_verdict == "DEBUNKED":
                final_label = "DEBUNKED"
                final_interpretation.extend([f"Metadata analysis indicates the content is false. Reason: {search_reason}", "Visual analysis results are overridden by these fact-checking findings."])
                colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"}
            elif visual_label == "Likely Deepfake" or (visual_label == "Suspicious" and search_verdict == "UNVERIFIED"):
                final_label = "LIKELY DEEPFAKE"
                final_interpretation.extend([f"Visual analysis detected strong indications of deepfake manipulation.", visual_result.get("interpretation", "")])
                colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"}
            elif search_verdict == "VERIFIED":
                final_label = "SOURCE VERIFIED"
                final_interpretation.append(f"The video's source is recognized as trusted and verified. Reason: {search_reason}.")
                colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"}
            elif visual_label == "Likely Real":
                final_label = "LIKELY REAL"
                final_interpretation.extend([f"Visual analysis indicates the video appears authentic with no signs of deepfake manipulation.", visual_result.get("interpretation", "")])
                colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"}
            else:
                final_label = visual_label.upper() 
                final_interpretation.append(f"Visual analysis found anomalies but could not conclusively determine manipulation. {visual_result.get('interpretation', '')}")

            return JSONResponse(content={
                "score_label": final_label, "classification_text": final_label, 
                "model_confidence": str(visual_conf), "scan_skipped": False,
                "colors": colors, "input_text": video_url, "interpretation": "\n".join(final_interpretation).strip(), 
                "suspicious_frames": visual_result.get("suspicious_frames", []), "search_verdict": search_verdict,
                "search_reason": search_reason, "metadata_confidence": metadata_confidence, "evidence": evidence_list, 
                "video_title": video_title, "timestamp": "Just Now", "timeline_graph": visual_result.get("timeline_graph"), 
                "frame_count": visual_result.get("frame_count", 0), "extracted_claim": extracted_claim 
            })
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    return router