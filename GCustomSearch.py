import os
import math
import re
import asyncio
import httpx
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBXn8fosKRpik2tYEevYvQnMc-np5a2WMM")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "f48bb9e7a0bf04a69")
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# --- DATA MODELS ---
class SearchResult(BaseModel):
    title: str
    link: str
    displayLink: str = ""
    snippet: str
    relevance_score: float = 0.0
    date: str = "N/A"
    matched_keywords: List[str] = []

class ScanRequest(BaseModel):
    article_text: str

class ScanResponse(BaseModel):
    verification_status: str
    dynamic_anchor_date: str
    extracted_claim: str
    verified_sources: List[SearchResult]

# --- CORE ENGINE (BM25 + DYNAMIC ANCHORING) ---
class CredibilityEngine:
    def __init__(self, k1=1.5, b=0.75, max_days=7):
        self.k1 = k1
        self.b = b
        self.max_days = max_days

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def extract_date(self, text: str) -> Optional[datetime]:
        # Relative Dates
        rel_match = re.search(r"(\d+)\s+(day|hour)s?\s+ago", text, re.IGNORECASE)
        if rel_match:
            val, unit = int(rel_match.group(1)), rel_match.group(2).lower()
            if unit == 'day': return datetime.now() - timedelta(days=val)
            if unit == 'hour': return datetime.now()
            
        # Absolute Dates
        date_regex = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}"
        match = re.search(date_regex, text, re.IGNORECASE)
        if not match: return None
        
        date_str = match.group().replace(',', '').replace('.', '')
        for fmt in ("%b %d %Y", "%B %d %Y", "%Y-%m-%d"):
            try: return datetime.strptime(date_str, fmt)
            except ValueError: continue
        return None

    def get_bm25_scores(self, query_tokens: List[str], corpus_tokens: List[List[str]]) -> List[float]:
        if not corpus_tokens: return []
        doc_lengths = [len(doc) for doc in corpus_tokens]
        avgdl = sum(doc_lengths) / len(corpus_tokens)
        df = Counter()
        for doc in corpus_tokens: df.update(set(doc))
        idf = {w: math.log(1 + (len(corpus_tokens) - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        
        scores = []
        for doc in corpus_tokens:
            score, counts = 0.0, Counter(doc)
            for word in query_tokens:
                if word in counts:
                    tf = counts[word]
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (len(doc) / avgdl))
                    score += idf.get(word, 0) * (num / den)
            scores.append(score)
        return scores

    async def fetch_web_evidence(self, query: str) -> List[dict]:
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": 10}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(SEARCH_URL, params=params)
                response.raise_for_status()
                return response.json().get("items", [])
            except Exception as e:
                print(f"Search API Error: {e}")
                return []

# --- MODULE WRAPPER FOR BACKWARD COMPATIBILITY ---
engine = CredibilityEngine()

async def fetch_supporting_articles(text_content: str, source_title: str = "", num_results: int = 10, match_threshold: float = 3.0) -> Tuple[List[SearchResult], List[str], str]:
    """
    Compatibility wrapper for analysis_helpers.py.
    Uses the new BM25 and Consensus logic with dynamic thresholds.
    """
    # 1-Sentence Claim extraction
    sentences = re.split(r'(?<=[.!?])\s+', text_content.strip())
    raw_claim = sentences[0] if sentences else ""
    
    # Remove datelines (e.g., "MANILA, PHILIPPINES — " or "CITY - ")
    claim = re.sub(r'^[A-Z\s,]+(?:\s[-—]\s?|\s?[-—]\s)', '', raw_claim).strip()
    if not claim: claim = raw_claim
    
    claim_tokens = engine.tokenize(claim)
    
    search_items = await engine.fetch_web_evidence(claim)
    if not search_items:
        return [], claim_tokens[:5], claim

    snippets = [item.get("snippet", "") for item in search_items]
    tokenized_corpus = [engine.tokenize(text_content)] + [engine.tokenize(s) for s in snippets]
    bm25_all_scores = engine.get_bm25_scores(claim_tokens, tokenized_corpus)
    bm25_scores = bm25_all_scores[1:] # Exclude the original text itself

    # Dynamic Anchor
    if bm25_scores:
        best_idx = bm25_scores.index(max(bm25_scores))
        dynamic_anchor = engine.extract_date(snippets[best_idx])
    else:
        dynamic_anchor = None

    verified_sources = []
    for i, item in enumerate(search_items):
        raw_snippet = item.get("snippet", "")
        doc_date = engine.extract_date(raw_snippet)
        
        # Date Weighting
        date_weight = 1.0
        if dynamic_anchor and doc_date:
            days_diff = abs((dynamic_anchor - doc_date).days)
            date_weight = max(0, 1 - (days_diff / engine.max_days))
        elif dynamic_anchor or doc_date:
            date_weight = 0.0
            
        final_score = bm25_scores[i] * date_weight
        
        # --- THE FIX: Uses the dynamic threshold parameter ---
        if final_score > match_threshold: 
            shared_keywords = list(set(claim_tokens).intersection(set(tokenized_corpus[i+1])))[:8]
            verified_sources.append(SearchResult(
                title=item.get("title", "Untitled"),
                link=item.get("link", "#"),
                displayLink=item.get("displayLink", "Source"),
                snippet=raw_snippet,
                relevance_score=round(final_score, 2),
                date=str(doc_date.date()) if doc_date else "N/A",
                matched_keywords=shared_keywords
            ))
            
    verified_sources.sort(key=lambda x: x.relevance_score, reverse=True)
    return verified_sources, claim_tokens[:6], claim
# --- FASTAPI APPLICATION ---
app = FastAPI(title="CredibilityScan Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/scan", response_model=ScanResponse)
async def scan_endpoint(request: ScanRequest):
    results, keywords, extracted_claim = await fetch_supporting_articles(request.article_text)
    
    # Extract cleaned claim again for the response metadata
    sentences = re.split(r'(?<=[.!?])\s+', request.article_text.strip())
    raw_claim = sentences[0] if sentences else ""
    claim = re.sub(r'^[A-Z\s,]+(?:\s[-—]\s?|\s?[-—]\s)', '', raw_claim).strip()
    if not claim: claim = raw_claim

    # Consensus Gate logic
    if len(results) < 2:
        status = "REJECTED / INSUFFICIENT CONSENSUS"
    else:
        status = f"VERIFIED BY CONSENSUS ({len(results)} Sources)"

    anchor_str = "NOT FOUND"
    if results:
        # Find original anchor date from the top result
        anchor_date = engine.extract_date(results[0].snippet)
        if anchor_date:
            anchor_str = str(anchor_date.date())

    return ScanResponse(
        verification_status=status,
        dynamic_anchor_date=anchor_str,
        extracted_claim=claim,
        verified_sources=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
