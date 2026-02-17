# routers/app_video.py

import os
import tempfile
import asyncio
import re
import string
import requests
import json 
import sys
from difflib import SequenceMatcher
from urllib.parse import urlparse
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse 
import yt_dlp
import deepfake_detector 
from GCustomSearch import fetch_supporting_articles

# =======================================================
# 1. CONFIGURATION
# =======================================================
API_KEY = "AIzaSyBXn8fosKRpik2tYEevYvQnMc-np5a2WMM" 
CX_ID = "f48bb9e7a0bf04a69"
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# TRUSTED CHANNEL MAPPING
TRUSTED_CHANNELS = {
    "gma": "GMA News",
    "abs-cbn": "ABS-CBN News",
    "abscbn": "ABS-CBN News",
    "rappler": "Rappler",
    "tv5": "News5",
    "news5": "News5",
    "inquirer": "Inquirer.net",
    "philstar": "Philstar",
    "manila times": "The Manila Times",
    "cnn philippines": "CNN Philippines",
    "ptv": "PTV Philippines",
    "anc": "ANC",
    "dzmm": "DZMM",
    "super radyo": "Super Radyo dzBB",
    "dzbb": "Super Radyo dzBB",
    "unang hirit": "Unang Hirit",
    "24 oras": "24 Oras",
    "saksi": "Saksi",
    "frontline pilipinas": "Frontline Pilipinas",
    "bandila": "Bandila",
    "tv patrol": "TV Patrol"
}

TRUSTED_DOMAINS = [
    "gmanetwork.com", "abs-cbn.com", "philstar.com", "inquirer.net", 
    "rappler.com", "verafiles.org", "manilatimes.net", "cnnphilippines.com", 
    "news.tv5.com.ph", "mb.com.ph", "pna.gov.ph", "sunstar.com.ph", 
    "abante.com.ph", "bbc.com", "reuters.com", "dw.com", "aljazeera.com",
    "youtube.com"
]

STOPWORDS = {
    "akin", "aking", "ako", "alin", "am", "amin", "aming", "ang", "ano", "anumang", "apat", "at", "atin", "ating", "ay", 
    "bababa", "bago", "bakit", "bawat", "bilang", "dahil", "dalawadapat", "din", "dito", "doon", "gagawin", "gayunman", 
    "ginagawa", "ginawa", "ginawang", "gumawa", "gusto", "habang", "hanggang", "hindi", "huwag", "iba", "ibaba", "ibabaw", 
    "ibig", "ikaw", "ilagay", "ilalim", "ilan", "inyong", "isa", "isang", "itaas", "ito", "iyo", "iyon", "iyong", "ka", 
    "kahit", "kailangan", "kailanman", "kami", "kanila", "kanilang", "kanino", "kanya", "kanyang", "kapag", "kapwa", 
    "karamihan", "katiyakan", "katulad", "kaya", "kaysa", "ko", "kong", "kulang", "kumuha", "kung", "laban", "lahat", 
    "lamang", "likod", "lima", "maaari", "maaaring", "maging", "mahusay", "makita", "marami", "marapat", "masyadom", 
    "may", "mayroon", "mga", "minsan", "mismo", "mula", "muli", "na", "nabanggit", "naging", "nagkaroon", "nais", 
    "nakita", "namin", "napaka", "narito", "nasaan", "ng", "ngayon", "ni", "nila", "nilang", "nito", "niya", "niyang", 
    "noon", "o", "pa", "paano", "pababa", "paggawa", "pagitan", "pagkakaroon", "pagkatapos", "palabas", "pamamagitan", 
    "panahon", "pangalawa", "para", "paraan", "pareho", "pataas", "pero", "pumunta", "pumupunta", "sa", "saan", "sabi", 
    "sabihin", "sarili", "sila", "sino", "siya", "tatlo", "tayo", "tulad", "tungkol", "una", "walang", "nang", "si",
    "po", "ho", "nyo", "n'yo", "d'yan", "sakin", "saakin", "sayang", "natin", "atin", "kayo", "sayo", "kanila",
    "nakikitang", "daw", "raw", "ba", "kase", "niyo", "nyo", "kayong", "silang", "vs", "versus",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "video", 
    "watch", "full", "hd", "official", "viral", "trending", "live", "exclusive", "breaking", "news"
}

# =======================================================
# 2. SILENT LOGGER
# =======================================================
class QuietLogger:
    def debug(self, msg): pass 
    def warning(self, msg): pass 
    def error(self, msg): print(msg) 

# =======================================================
# 3. EXTRACTION HELPERS
# =======================================================

def get_clean_tokens(text):
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    tokens = [w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2]
    return set(tokens)

def extract_keywords(title, tags=[]):
    print(f"   [Metadata] 1. Raw Title: {title}", flush=True)
    # 1. CHOPPING
    chopped_title = title
    if "|" in title: chopped_title = title.split("|")[0]
    if " - " in title: chopped_title = title.split(" - ")[0]

    # 2. REGEX CLEANING
    clean_title = re.sub(r"\[.*?\]", "", chopped_title) 
    clean_title = re.sub(r"\(.*?\)", "", clean_title)
    clean_title = re.sub(r"#\S+", "", clean_title) 
    clean_title = re.sub(r"(?i)^(panoorin|watch|live|exclusive|breaking)\s*", "", clean_title.strip())

    # 3. STOPWORD REMOVAL
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = clean_title.translate(translator).lower()
    tokens = [w for w in text_no_punct.split() if w not in STOPWORDS and len(w) > 2]
    
    # 4. FALLBACK
    if len(tokens) < 3 and tags:
        tokens.extend([t.lower() for t in tags[:3]])

    final_query = " ".join(tokens)
    if not final_query.strip(): final_query = title

    print(f"   [Metadata] 3. Final Query:    {final_query}", flush=True)
    return final_query

def calculate_keyword_coverage(video_title, result_title):
    vid_tokens = get_clean_tokens(video_title)
    res_tokens = get_clean_tokens(result_title)
    if not vid_tokens: return 0.0
    matching_words = vid_tokens.intersection(res_tokens)
    return len(matching_words) / len(vid_tokens)

def check_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_website_name(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except:
        return "Unknown Site"

def scrape_tiktok_oembed(video_url):
    try:
        print(f"   [Fallback] Attempting TikTok oEmbed for: {video_url}", flush=True)
        oembed_url = f"https://www.tiktok.com/oembed?url={video_url}"
        response = requests.get(oembed_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "TikTok Video"),
                "author": data.get("author_name", "TikTok User"),
                "thumbnail": data.get("thumbnail_url", ""),
                "platform": "TikTok",
                "tags": [] 
            }
    except Exception as e:
        print(f"   [Fallback] TikTok oEmbed failed: {e}", flush=True)
    return None

async def analyze_metadata_credibility(video_title, video_tags=[], video_author=""):
    print(f"\n=======================================================", flush=True)
    print(f"   [Metadata] Analysis Started. Author: {video_author}", flush=True)

    is_trusted_channel = False
    trusted_reason = ""
    clean_author = video_author.lower().strip()
    
    # 1. Check if channel is in Trusted List
    for key, official_name in TRUSTED_CHANNELS.items():
        if key in clean_author:
            print(f"   [Metadata] Trusted Channel Detected: {official_name}", flush=True)
            is_trusted_channel = True
            trusted_reason = f"Source verified as official {official_name} channel."
            break 

    # Fallback claim extraction if search is skipped
    sentences = re.split(r'(?<=[.!?])\s+', video_title.strip())
    raw_claim = sentences[0] if sentences else ""
    extracted_claim = re.sub(r'^[A-Z\s,]+(?:\s[-—]\s?|\s?[-—]\s)', '', raw_claim).strip()
    if not extracted_claim: extracted_claim = raw_claim

    # --- SKIP SEARCH IF UNVERIFIED ---
    if not is_trusted_channel:
        print(f"   [Metadata] Unverified Source ({video_author}). Skipping Search to save quota.", flush=True)
        return "UNVERIFIED", 20, [], "Source not in trusted database. External search skipped.", extracted_claim

# 2. Proceed to Google Search via GCustomSearch
    print(f"   [Metadata] Trusted source detected. Verifying content via GCustomSearch...", flush=True)
    evidence_list = []
    
    try:
        # THE FIX: We pass match_threshold=1.2 to account for shorter video titles
        results, keywords, extracted_claim = await fetch_supporting_articles(video_title, match_threshold=1.0)
        
        for item in results:
            result_title = item.title
            link = item.link
            snippet = item.snippet
            
            is_trusted_domain = any(trusted in link.lower() for trusted in TRUSTED_DOMAINS)
            website = urlparse(link).netloc.replace("www.", "")
            status = "RELATED"
            
            if is_trusted_domain:
                status = "VERIFIED"
            
            if any(x in result_title.lower() for x in ["false", "hoax", "fake", "fact check", "misleading", "hindi totoo"]):
                status = "DEBUNKED"
            
            # Map GCustomSearch structure to the Video structure
            evidence_list.append({
                "title": result_title,
                "link": link,
                "website": website,
                "displayLink": item.displayLink,
                "status": status,
                "snippet": snippet,
                "relevance_score": item.relevance_score,
                "matched_keywords": item.matched_keywords
            })

    except Exception as e:
        print(f"   [!] GCustomSearch Error: {e}", flush=True)

    # FINAL DECISION LOGIC
    metadata_confidence = 0 
    if any(e['status'] == "DEBUNKED" for e in evidence_list):
        metadata_confidence = 95 
        return "DEBUNKED", metadata_confidence, evidence_list, "Flagged by fact-checkers as false.", extracted_claim

    if is_trusted_channel:
        metadata_confidence = 90 
        return "VERIFIED", metadata_confidence, evidence_list, trusted_reason, extracted_claim
        
    if any(e['status'] == "VERIFIED" for e in evidence_list):
        metadata_confidence = 85 
        return "VERIFIED", metadata_confidence, evidence_list, "Confirmed by trusted news sources.", extracted_claim
    elif evidence_list:
        metadata_confidence = 60 
        return "CORROBORATED", metadata_confidence, evidence_list, "Similar content found online.", extracted_claim
    else:
        metadata_confidence = 20 
        return "UNVERIFIED", metadata_confidence, [], "No matching reports found from trusted sources.", extracted_claim
# =======================================================
# 4. API ROUTES
# =======================================================
def create_video_router(deepfake_model=None, yolo_model=None, **kwargs):
    
    router = APIRouter()

    @router.post("/fetch-video-metadata")
    async def fetch_video_metadata(video_url: str = Form(...)):
        ydl_opts = {'quiet': True, 'noplaylist': True, 'skip_download': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                author_name = info.get('uploader') or info.get('channel') or info.get('uploader_id') or "Unknown Source"
                return JSONResponse({
                    "status": "success",
                    "title": info.get('title', 'Unknown Title'),
                    "thumbnail": info.get('thumbnail', ''),
                    "duration": info.get('duration', 0),
                    "author": author_name, 
                    "platform": info.get('extractor_key', 'Unknown Platform'),
                    "tags": info.get('tags', [])
                })
        except Exception as e:
            if "tiktok.com" in video_url:
                fb = scrape_tiktok_oembed(video_url)
                if fb: return JSONResponse({"status": "success", **fb, "duration": 0})
            return JSONResponse({"status": "error", "message": f"Could not fetch metadata: {str(e)}"}, status_code=400)

    @router.post("/analyze-video")
    async def analyze_video(request: Request, video_url: str = Form(...)):
        downloaded_file_path = None

        visual_result = {"verdict": "Uncertain", "confidence": 0, "suspicious_frames": []}
        # Extract visual verdict and confidence from visual_result
        visual_label = visual_result.get("verdict", "Uncertain")
        visual_conf = visual_result.get("confidence", 0)

        try:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, 'scan_%(id)s.%(ext)s')
            
            # --- STRICT LIMITS ---
            MAX_DURATION_SECONDS = 180  # 3 Minutes
            MAX_SIZE_MB = 100           # 100 MB Limit
            MAX_BYTES = MAX_SIZE_MB * 1024 * 1024

            def check_limits(info, *, incomplete=False):
                duration = info.get('duration')
                if duration and duration > MAX_DURATION_SECONDS: return 'Video is too long'
                filesize = info.get('filesize') or info.get('filesize_approx')
                if filesize and filesize > MAX_BYTES: return 'Video is too large'
                return None

            def stop_on_large_download(d):
                if d.get('downloaded_bytes', 0) > MAX_BYTES: raise Exception("larger than max-filesize")

            ydl_opts = {
                'outtmpl': temp_path,
                'format': 'best[ext=mp4]/best', 
                'match_filter': check_limits,
                'progress_hooks': [stop_on_large_download],
                'noplaylist': True, 'retries': 3, 'socket_timeout': 15, 'quiet': True
            }
            
            # 1. Metadata Fetch
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

            # 2. Attempt Download
            download_error_msg = None
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    downloaded_file_path = ydl.prepare_filename(info)
            except Exception as e:
                download_error_msg = str(e)

            # 3. CHECK: Did it actually download?
            if download_error_msg or not downloaded_file_path or not os.path.exists(downloaded_file_path):
                
                # Run Metadata Search (Now Awaited and returning 5 variables)
                search_verdict, metadata_confidence, evidence_list, search_reason, extracted_claim = await analyze_metadata_credibility(video_title, video_tags, video_author)
                
                # Determine Label
                final_label = "UNKNOWN SOURCE"
                colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"} 

                if search_verdict == "VERIFIED":
                    final_label = "SOURCE VERIFIED"
                    colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"} 
                elif search_verdict == "DEBUNKED":
                    final_label = "DEBUNKED"
                    colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"} 

                note_message = "<b>Note:</b> Video scan skipped (Download failed). Metadata verified below."
                err_lower = (download_error_msg or "video is too long").lower()

                if "too long" in err_lower or "max-filesize" in err_lower or "too large" in err_lower:
                    note_message = "<b>Note:</b> Video exceeded maximum limit (length/size). However, we checked the source metadata above."

                return JSONResponse(content={
                    "score_label": final_label,
                    "classification_text": "Visual Scan Skipped",
                    "model_confidence": None, 
                    "scan_skipped": True,    
                    "colors": colors,
                    "input_text": video_url,
                    "interpretation": note_message,
                    "suspicious_frames": [],
                    "search_verdict": search_verdict,
                    "search_reason": search_reason,
                    "metadata_confidence": metadata_confidence, 
                    "evidence": evidence_list, 
                    "video_title": video_title,
                    "timestamp": "Just Now",
                    "extracted_claim": extracted_claim # <--- PASS TO FRONTEND
                })
            # --- UPDATED: SEQUENTIAL SCAN TO PASS 'is_verified' TO DETECTOR ---
# --- UPDATED: SEQUENTIAL SCAN TO PASS 'is_verified' TO DETECTOR ---
            async def run_search():
                # Await the new async function and capture 5 variables
                return await analyze_metadata_credibility(video_title, video_tags, video_author)

            # 4. We need to run search FIRST to get the 'VERIFIED' status
            search_verdict, metadata_confidence, evidence_list, search_reason, extracted_claim = await run_search()
            is_verified = (search_verdict == "VERIFIED")

            # 5. Then run the visual scan WITH the verified flag
            if deepfake_model and yolo_model:
                visual_result = deepfake_detector.run_deepfake_analysis(
                    downloaded_file_path, 
                    yolo_model, 
                    deepfake_model, 
                    is_verified_source=is_verified # <--- THE FIX
                )
            else:
                visual_result = {"verdict": "Uncertain", "confidence": 0, "suspicious_frames": []}

            # Re-extract visual_label and visual_conf after deepfake_detector.run_deepfake_analysis
            visual_label = visual_result.get("verdict", "Uncertain")
            visual_conf = visual_result.get("confidence", 0)

            print(f"DEBUG: visual_label: {visual_label}, search_verdict: {search_verdict}", flush=True)

            final_label = "Uncertain" # Default
            final_interpretation = []
            colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"} # Default Yellow

            # Hierarchy of Verdicts:
            # 1. DEBUNKED (Highest Priority)
            if search_verdict == "DEBUNKED":
                final_label = "DEBUNKED"
                final_interpretation.append(f"Metadata analysis indicates the content is false. Reason: {search_reason}")
                final_interpretation.append("Visual analysis results are overridden by these fact-checking findings.")
                colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"} # Red
            
            # 2. LIKELY DEEPFAKE (Based on visual result)
            elif visual_label == "Likely Deepfake" or (visual_label == "Suspicious" and search_verdict == "UNVERIFIED"):
                final_label = "LIKELY DEEPFAKE"
                final_interpretation.append(f"Visual analysis detected strong indications of deepfake manipulation.")
                final_interpretation.append(visual_result.get("interpretation", ""))
                if search_verdict == "UNVERIFIED":
                    final_interpretation.append(f"Warning: The video's source could not be verified, adding to the overall suspicion.")
                colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"} # Red

            # 3. SOURCE VERIFIED (Strong positive signal)
            elif search_verdict == "VERIFIED":
                final_label = "SOURCE VERIFIED"
                final_interpretation.append(f"The video's source is recognized as trusted and verified. Reason: {search_reason}.")
                if visual_label in ["Visual Anomalies Detected (Source Verified)", "Likely Real (Noise Detected)"]:
                    final_interpretation.append(f"Visual analysis detected some minor artifacts, but these are considered noise due to the high credibility of the source.")
                else:
                    final_interpretation.append(f"Visual analysis did not detect strong deepfake signals and aligns with the verified source status.")
                colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"} # Green
            
            # 4. LIKELY REAL (Based on visual result)
            elif visual_label == "Likely Real":
                final_label = "LIKELY REAL"
                final_interpretation.append(f"Visual analysis indicates the video appears authentic with no signs of deepfake manipulation.")
                final_interpretation.append(visual_result.get("interpretation", ""))
                if search_verdict == "CORROBORATED":
                    final_interpretation.append(f"Metadata analysis found corroborating articles, supporting the authenticity.")
                elif search_verdict == "UNVERIFIED":
                    final_interpretation.append(f"Note: The video's source remains unverified, but visual analysis shows no deepfake signs.")
                colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"} # Green

            # 5. SUSPICIOUS / INCONCLUSIVE (Intermediate signals)
            elif visual_label in ["Suspicious", "Uncertain"] or visual_label == "Visual Anomalies Detected (Source Verified)":
                final_label = visual_label.upper() # Use the specific label from detector
                final_interpretation.append(f"Visual analysis found anomalies but could not conclusively determine manipulation. {visual_result.get('interpretation', '')}")
                
                if visual_label == "Visual Anomalies Detected (Source Verified)":
                    final_interpretation.append(f"Despite visual red flags, the video's source is verified, leading to a 'Visual Anomalies Detected' classification rather than 'Deepfake'.")
                elif search_verdict == "UNVERIFIED":
                    final_interpretation.append(f"Additionally, the video's source could not be verified, adding to the overall uncertainty.")
                elif search_verdict == "CORROBORATED":
                    final_interpretation.append(f"Metadata analysis found some corroborating articles, but visual analysis remains suspicious.")
                
                colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"} # Yellow

            # 6. UNKNOWN SOURCE (Fallback)
            else:
                final_label = "UNKNOWN SOURCE"
                final_interpretation.append("The system could not make a definitive visual or metadata classification for this video.")
                if search_reason:
                    final_interpretation.append(f"Metadata analysis reason: {search_reason} (Confidence: {metadata_confidence}%).")
                else:
                    final_interpretation.append(f"No specific reason from metadata analysis (Confidence: {metadata_confidence}%).")
                if visual_result.get("interpretation"):
                    final_interpretation.append(f"Visual analysis interpretation: {visual_result['interpretation']} (Confidence: {visual_conf}%).")
                colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"} # Yellow

            return JSONResponse(content={
                "score_label": final_label,
                "classification_text": final_label, 
                "model_confidence": str(visual_conf),
                "scan_skipped": False,
                "colors": colors,
                "input_text": video_url,
                "interpretation": "\n".join(final_interpretation).strip(), 
                "suspicious_frames": visual_result.get("suspicious_frames", []),
                "search_verdict": search_verdict,
                "search_reason": search_reason,
                "metadata_confidence": metadata_confidence, 
                "evidence": evidence_list, 
                "video_title": video_title,
                "timestamp": "Just Now",
                "timeline_graph": visual_result.get("timeline_graph"), 
                "frame_count": visual_result.get("frame_count", 0),
                "extracted_claim": extracted_claim # <--- PASS TO FRONTEND
            })
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
        finally:
            if downloaded_file_path and os.path.exists(downloaded_file_path):
                try: os.remove(downloaded_file_path)
                except: pass
    
    return router