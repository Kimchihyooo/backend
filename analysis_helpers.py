# analysis_helpers.py
# (FINAL VERSION: Highlights Top 3 Influential Sentences in RED)

import langid
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Optional, Tuple
import re
import string
import numpy as np 
import scipy.sparse 

# Custom imports
from model_loader import get_model_loader
from GCustomSearch import fetch_supporting_articles
from stopwords import filipino_stopwords, english_stopwords

# ===================================================================
# HELPER: Extract Sentiment
# ===================================================================
def extract_sentiment_for_app(text: str, pipeline) -> List[int]:
    try:
        result = pipeline(str(text)[:512])[0]
        label = result['label'].lower()
        pos = 1 if label == 'positive' else 0
        neg = 1 if label == 'negative' else 0
        neut = 1 if label == 'neutral' else 0
        return [pos, neg, neut]
    except Exception as e:
        print(f"Sentiment Error: {e}")
        return [0, 0, 1] 

# ===================================================================
# HELPER: Calculate Sentence Scores
# ===================================================================
def calculate_sentence_scores(text: str, lime_list: List[tuple]) -> Tuple[List[Dict], List[str]]:
    """
    Optimized function to split text into sentences and calculate a score for each
    based on LIME weights using set intersection for efficiency.
    Returns:
      1. List of dicts with scores (sorted by influence, top 3 only)
      2. List of text parts (sentences + separators) for HTML reconstruction
    """
    if not lime_list:
        return [], [text]

    # 1. Implement Pre-Filtering on the weight_map to exclude any word with a weight of exactly zero.
    # Also convert to a set for faster lookups during intersection
    filtered_weight_map = {word.lower(): weight for word, weight in lime_list if weight != 0}
    
    # Split by (Punctuation+Whitespace) OR (Newlines) to keep structure
    parts = re.split(r'([.!?]+\s+|\n+)', text)
    
    scored_sentences = []
    
    # Iterate over text parts (even indices) which are sentences
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if not sent.strip():
            continue
            
        # Tokenize sentence words and convert to a set for intersection
        sentence_words = set(re.findall(r'\b\w+\b', sent.lower()))
        
        # Calculate intersection of words in sentence and influential words
        # and sum their LIME weights efficiently
        sent_score = sum(filtered_weight_map[word] for word in sentence_words.intersection(filtered_weight_map))
        
        scored_sentences.append({
            "sentence": sent,
            "score": sent_score,
            "abs_score": abs(sent_score) # Used for ranking influence
        })
        
    # Sort by Absolute Score (Most Influential first)
    top_sentences = sorted(scored_sentences, key=lambda x: x["abs_score"], reverse=True)
    
    # 2. Change the return logic to strictly provide only the Top 3 most influential sentences.
    final_top_list = []
    
    # Calculate the sum of absolute scores for the actual top 3 sentences for percentage calculation
    # This ensures percentages are relative to only the top 3 chosen for display.
    total_abs_for_pct = sum(s["abs_score"] for s in top_sentences[:3])

    # Only iterate through the top 3 sentences
    for s in top_sentences[:3]: 
        pct = (s["abs_score"] / total_abs_for_pct * 100) if total_abs_for_pct > 0 else 0
        
        final_top_list.append({
            "sentence": s["sentence"],
            "score": s["score"],
            "abs_score": s["abs_score"],
            "percentage": round(pct, 2)
        })
        
    return final_top_list, parts

# ===================================================================
# HELPER: Generate HTML (UPDATED LOGIC: TOP 3 WITH PREMIUM TOOLTIP)
# ===================================================================
def generate_sentence_highlight_html(text_parts: List[str], top_sentences: List[Dict]) -> str:
    """
    Highlights the Top 3 influential sentences in RED and adds a custom hover tooltip.
    """
    # Create a map of the Top 3 sentences for fast lookup
    # We take [:3] to restrict highlighting to only the top 3
    top_3_map = {s['sentence']: s for s in top_sentences[:3]}
    
    html_output = []
    
    # Iterate through parts (Even = Sentence, Odd = Separator)
    for i, part in enumerate(text_parts):
        if i % 2 != 0: 
            # Separator handling (newlines to <br>)
            if '\n' in part:
                html_output.append("<br>")
            else:
                html_output.append(part)
            continue
            
        sent = part
        if not sent.strip():
            html_output.append(sent)
            continue

        # CHECK: Is this sentence in the Top 3?
        if sent in top_3_map:
            data = top_3_map[sent]
            # HIGHLIGHT RED with Premium CSS Tooltip Structure
            html_output.append(
                f'<span class="lime-bad inline-tooltip-container">'
                f'{sent}'
                f'<span class="inline-tooltip-box">'
                f'Sentences highlighted in red indicate key linguistic features and patterns that most heavily influenced the model\'s \'Fake\' classification.'
                f'</span>'
                f'</span>'
            )
        else:
            # Normal text (No highlight)
            html_output.append(sent)

    return "".join(html_output)

# ===================================================================
# HELPER: LIME Explanation Generator (Standard)
# ===================================================================
def get_lime_explanation(text: str, lang: str, models: Dict, predicted_label: int) -> Optional[List[tuple]]:
    try:
        explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
        
        # Define predict_proba_complex based on language
        predict_proba_complex = None

        if lang == 'en':
            xgb_model = models.get("xgb_model")
            loader = get_model_loader()
            sentiment_pipe = models.get("sentiment_pipeline")
            stopwords_list = english_stopwords # Keep for potential filtering if needed later, though not directly used in embedding prediction
            
            if not xgb_model or not loader:
                print("Error: English XGBoost model or loader not available for LIME.")
                return None

            def predict_proba_complex_en(texts):
                embeddings = []
                for t in texts:
                    embedding = loader.article_to_embedding(t)
                    embeddings.append(embedding)
                return xgb_model.predict_proba(np.array(embeddings))
            predict_proba_complex = predict_proba_complex_en

        elif lang == 'tl':
            vectorizer = models.get("vectorizer_tl")
            model = models.get("model_tl")
            sentiment_pipe = models.get("sentiment_pipeline")
            stopwords_list = filipino_stopwords

            if not vectorizer or not model:
                print("Error: Tagalog TF-IDF model or vectorizer not available for LIME.")
                return None

            def predict_proba_complex_tl(texts):
                tfidf_vecs = vectorizer.transform(texts)
                combined_list = []
                for t in texts:
                    if sentiment_pipe:
                        sent_vec = extract_sentiment_for_app(t, sentiment_pipe)
                    else:
                        sent_vec = [0, 0, 1]
                    combined_list.append(sent_vec)
                sentiment_matrix = np.array(combined_list)
                combined_features = scipy.sparse.hstack([tfidf_vecs, sentiment_matrix])
                return model.predict_proba(combined_features)
            predict_proba_complex = predict_proba_complex_tl
        else:
            print(f"Error: Language '{lang}' not supported for LIME explanation.")
            return None

        # Ensure predict_proba_complex is set before calling explainer
        if predict_proba_complex is None:
            return None

        print(f"Running LIME for '{lang}'...")
        exp = explainer.explain_instance(text, predict_proba_complex, num_features=50, num_samples=100, labels=[predicted_label])
        return exp.as_list()

    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        return None

# ===================================================================
# HELPER: Detect Bias (ROBUST VERSION)
# ===================================================================
def detect_bias(text: str, lang: str, models: Dict) -> Dict:
    """
    Detects political bias and returns ALL probabilities.
    """
    try:
        if lang == 'en':
            vectorizer = models.get("bias_vectorizer_en")
            model = models.get("bias_model_en")
        elif lang == 'tl':
            vectorizer = models.get("bias_vectorizer_tl")
            model = models.get("bias_model_tl")
        else:
            return None
            
        if not vectorizer or not model:
            return None

        # Predict
        tfidf_vec = vectorizer.transform([text])
        probabilities = model.predict_proba(tfidf_vec)[0]
        classes = model.classes_ 
        
        # Build dictionary of all scores with STANDARDIZED keys
        all_scores = {"Left": 0, "Center": 0, "Right": 0}
        
        for cls, prob in zip(classes, probabilities):
            # Clean label name
            raw_label = str(cls).replace("['", "").replace("']", "").replace('["', '').replace('"]', '')
            label_lower = raw_label.lower()
            
            # Map to standardized keys
            percent = round(prob * 100, 1)
            if "left" in label_lower:
                all_scores["Left"] = percent
            elif "right" in label_lower:
                all_scores["Right"] = percent
            elif "center" in label_lower:
                all_scores["Center"] = percent
            else:
                # Fallback if label is weird (e.g. "0", "1") - Adjust if needed
                all_scores[raw_label] = percent

        # Get Dominant Label based on the new standardized dict
        dominant_label = max(all_scores, key=all_scores.get)
        dominant_conf = all_scores[dominant_label]

        return {
            "label": dominant_label, 
            "confidence": dominant_conf,
            "all_scores": all_scores 
        }
    except Exception as e:
        print(f"Bias Detection Error: {e}")
        return None
# ===================================================================
# MAIN FUNCTION
# ===================================================================
async def analyze_text_content(text: str, models: Dict) -> Dict:
    
    # 1. Detect Language
    try:
        lang, _ = langid.classify(text)
        print(f"Detected language: {lang}")
    except Exception as e:
        return {"error": "Could not detect language."}

    prediction = [0]
    probability = np.array([[0.5, 0.5]]) 
    
    # 2. Setup Variables & 3. Predict
    if lang == 'en':
        xgb_model = models.get("xgb_model")
        loader = get_model_loader()
        stopwords_list = english_stopwords # Keep for LIME filtering if needed
        
        if not xgb_model or not loader:
            return {"error": "English XGBoost model or loader not available."}
        
        try:
            embedding = loader.article_to_embedding(text)
            probability = xgb_model.predict_proba([embedding])
            prediction_val = 1 if probability[0][1] >= 0.50 else 0
            prediction = [prediction_val]
        except Exception as e:
            return {"error": f"Error during English analysis (embedding/prediction): {e}"}

    elif lang == 'tl':
        vectorizer = models.get("vectorizer_tl")
        model = models.get("model_tl")
        stopwords_list = filipino_stopwords
        sentiment_pipe = models.get("sentiment_pipeline") # Moved here for TL specific use
        
        if not vectorizer or not model:
            return {"error": "Tagalog TF-IDF model or vectorizer not available."}
            
        try:
            tfidf_vec = vectorizer.transform([text])
            if sentiment_pipe:
                sentiment_vec = extract_sentiment_for_app(text, sentiment_pipe)
            else:
                sent_vec = [0, 0, 1]
            
            combined_features = scipy.sparse.hstack([tfidf_vec, np.array([sentiment_vec])])
            probability = model.predict_proba(combined_features)
            prediction = model.predict(combined_features)
            
        except Exception as e:
            return {"error": f"Error during Tagalog analysis: {e}"}
    else:
        return {"error": f"Language '{lang}' is not supported."}

    # 4. Results
    prediction_val = int(prediction[0])
    classification_label = "FAKE" if prediction_val == 1 else "REAL"
    model_confidence = probability[0][prediction_val]
    model_confidence_percent = round(model_confidence * 100)
    
    # 5. LIME & Sentence Analysis
    influential_sentences = []
    sentence_html = text 
    lime_explanation_filtered = []

    if classification_label == "FAKE":
        lime_explanation = get_lime_explanation(text, lang, models, prediction_val)
        if lime_explanation:
            # A. Calculate influence scores
            influential_sentences, text_parts = calculate_sentence_scores(text, lime_explanation)
            
            # B. Generate HTML (Highlight Top 3 in RED)
            sentence_html = generate_sentence_highlight_html(text_parts, influential_sentences)
            
            # C. Filter LIME list for JSON return
            lime_explanation_filtered = [
                (word, weight) for word, weight in lime_explanation
                if word.lower() not in stopwords_list
            ][:10]

    # --- NEW: Run Bias Detection ---
    bias_data = detect_bias(text, lang, models)

   # 6. Search Logic (UNIFIED FOR ALL CASES)
    fetched_articles, keywords, _ = await fetch_supporting_articles(text)
    supporting_articles = [] 
    
    if keywords:
        keyword_set = set(keywords)
        for article in fetched_articles:
            title_lower = getattr(article, "title", "").lower()
            if any(k in title_lower for k in keyword_set):
                supporting_articles.append(article)
                
    if not supporting_articles:
        supporting_articles = fetched_articles[:5] 
        
    link_count = len(supporting_articles)
    
    # Extract cleaned claim for UI
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    raw_claim = sentences[0] if sentences else ""
    extracted_claim = re.sub(r'^[A-Z\s,]+(?:\s[-—]\s?|\s?[-—]\s)', '', raw_claim).strip()
    if not extracted_claim: extracted_claim = raw_claim

    explanation = ""

    # === APPLICATION OF THE 4-CASE LOGIC MATRIX ===
    if classification_label == "REAL":
        if link_count > 0:
            # Case 1: Real News + Evidence
            Category_Label = "MOST LIKELY REAL"
            colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"}
            explanation = "Detected linguistic patterns typical of real news, and verified external sources are actively reporting this claim."
        else:
            # Case 2: Real News + NO Evidence
            Category_Label = "UNCERTAIN"
            colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"}
            explanation = "Detected linguistic patterns typical of real news, but could not find any verified external sources reporting this claim. Please verify independently."
    else: # classification_label == "FAKE"
        if link_count > 0:
            # Case 3: Fake News + Evidence
            Category_Label = "UNCERTAIN"
            colors = {"text_color": "#a16207", "bg_color": "#fefce8", "accent_color": "#f59e0b"}
            explanation = "Detected highly manipulative or unnatural linguistic patterns typical of fake news, but found external sources discussing this topic. It may be heavily biased, sensationalized, or a real event reported poorly."
        else:
            # Case 4: Fake News + NO Evidence
            Category_Label = "MOST LIKELY FAKE"
            colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"}
            explanation = "The AI detected deceptive linguistic patterns, and no credible external sources could be found to verify this claim."

    # 7. Serialization
    serializable_articles = []
    if supporting_articles:
        for article in supporting_articles:
            serializable_articles.append({
                "title": getattr(article, "title", "No Title"),
                "link": getattr(article, "link", "#"),
                "displayLink": getattr(article, "displayLink", "Source"),
                "snippet": getattr(article, "snippet", ""),
                "relevance_score": getattr(article, "relevance_score", 0),
                "matched_context": getattr(article, "matched_context", ""),
                "matched_keywords": getattr(article, "matched_keywords", [])
            })
            
    # 8. Return
    return {
        "model_confidence": str(model_confidence_percent),
        "score_label": Category_Label,
        "classification_text": classification_label,
        "colors": colors,
        "supporting_articles": serializable_articles, 
        "news_text": text,
        "extracted_claim": extracted_claim,
        "lime_explanation": lime_explanation_filtered, 
        "lime_html": sentence_html, 
        "influential_sentences": influential_sentences,
        "bias_data": bias_data,
        "explanation": explanation  # <-- Pass the new explanation to the frontend
    }