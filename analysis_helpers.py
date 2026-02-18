# analysis_helpers.py
# (OPTIMIZED VERSION: Removed Sentiment Pipelines for RoBERTa+XGBoost Workflow)

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
# HELPER: Calculate Sentence Scores
# ===================================================================
def calculate_sentence_scores(text: str, lime_list: List[tuple]) -> Tuple[List[Dict], List[str]]:
    """
    Optimized function to split text into sentences and calculate a score for each
    based on LIME weights using set intersection for efficiency.
    """
    if not lime_list:
        return [], [text]

    filtered_weight_map = {word.lower(): weight for word, weight in lime_list if weight != 0}
    parts = re.split(r'([.!?]+\s+|\n+)', text)
    scored_sentences = []
    
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if not sent.strip():
            continue
            
        sentence_words = set(re.findall(r'\b\w+\b', sent.lower()))
        sent_score = sum(filtered_weight_map[word] for word in sentence_words.intersection(filtered_weight_map))
        
        scored_sentences.append({
            "sentence": sent,
            "score": sent_score,
            "abs_score": abs(sent_score) 
        })
        
    top_sentences = sorted(scored_sentences, key=lambda x: x["abs_score"], reverse=True)
    final_top_list = []
    total_abs_for_pct = sum(s["abs_score"] for s in top_sentences[:3])

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
# HELPER: Generate HTML (TOP 3 HIGHLIGHTS)
# ===================================================================
def generate_sentence_highlight_html(text_parts: List[str], top_sentences: List[Dict]) -> str:
    """
    Highlights the Top 3 influential sentences in RED and adds a custom hover tooltip.
    """
    top_3_map = {s['sentence']: s for s in top_sentences[:3]}
    html_output = []
    
    for i, part in enumerate(text_parts):
        if i % 2 != 0: 
            if '\n' in part:
                html_output.append("<br>")
            else:
                html_output.append(part)
            continue
            
        sent = part
        if not sent.strip():
            html_output.append(sent)
            continue

        if sent in top_3_map:
            html_output.append(
                f'<span class="lime-bad inline-tooltip-container">'
                f'{sent}'
                f'<span class="inline-tooltip-box">'
                f'Sentences highlighted in red indicate key linguistic features and patterns that most heavily influenced the model\'s \'Fake\' classification.'
                f'</span>'
                f'</span>'
            )
        else:
            html_output.append(sent)

    return "".join(html_output)

# ===================================================================
# HELPER: LIME Explanation Generator (Embeddings Only)
# ===================================================================
def get_lime_explanation(text: str, lang: str, models: Dict, predicted_label: int) -> Optional[List[tuple]]:
    """
    Generates LIME explanations using text embeddings + XGBoost (English) 
    or TF-IDF (Tagalog). Sentiment features have been removed.
    """
    try:
        explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
        predict_proba_complex = None

        if lang == 'en':
            xgb_model = models.get("xgb_model")
            loader = get_model_loader()
            
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

            if not vectorizer or not model:
                print("Error: Tagalog TF-IDF model or vectorizer not available for LIME.")
                return None

            def predict_proba_complex_tl(texts):
                tfidf_vecs = vectorizer.transform(texts)
                # Removed sentiment concatenation here
                return model.predict_proba(tfidf_vecs)
            predict_proba_complex = predict_proba_complex_tl
        else:
            print(f"Error: Language '{lang}' not supported for LIME explanation.")
            return None

        if predict_proba_complex is None:
            return None

        print(f"Running LIME for '{lang}' (Embeddings-based)...")
        exp = explainer.explain_instance(
            text, 
            predict_proba_complex, 
            num_features=50, 
            num_samples=100, 
            labels=[predicted_label]
        )
        return exp.as_list()
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        return None

# ===================================================================
# HELPER: Detect Bias
# ===================================================================
def detect_bias(text: str, lang: str, models: Dict) -> Dict:
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

        tfidf_vec = vectorizer.transform([text])
        probabilities = model.predict_proba(tfidf_vec)[0]
        classes = model.classes_

        all_scores = {"Left": 0, "Center": 0, "Right": 0}
        for cls, prob in zip(classes, probabilities):
            label_lower = str(cls).lower()
            percent = round(prob * 100, 1)
            if "left" in label_lower: all_scores["Left"] = percent
            elif "right" in label_lower: all_scores["Right"] = percent
            elif "center" in label_lower: all_scores["Center"] = percent

        dominant_label = max(all_scores, key=all_scores.get)
        return {
            "label": dominant_label,
            "confidence": all_scores[dominant_label],
            "all_scores": all_scores
        }
    except Exception as e:
        print(f"Bias Detection Error: {e}")
        return None

# ===================================================================
# MAIN FUNCTION: Text Content Analysis
# ===================================================================
async def analyze_text_content(text: str, models: Dict) -> Dict:
    """
    Centralized function for full text credibility analysis.
    """
    # 1. Language Detection
    lang, _ = langid.classify(text)
    
    # 2. Extract Embedding and Predict via XGBoost
    loader = get_model_loader()
    embedding = loader.article_to_embedding(text)
    xgb_model = models.get("xgb_model")
    
    probs = xgb_model.predict_proba(embedding.reshape(1, -1))[0]
    predicted_label = int(np.argmax(probs))
    model_confidence = float(np.max(probs))
    
    # 3. LIME and Bias
    lime_explanation = get_lime_explanation(text, lang, models, predicted_label)
    bias_data = detect_bias(text, lang, models)
    
    # 4. External Search Verification
    # (Placeholder: Extract a core claim from the first sentence)
    claim = text.split('.')[0]
    supporting_articles, _, extracted_claim = await fetch_supporting_articles(claim)

    # 5. Build Result (Standard format for your frontend)
    Category_Label = "FAKE" if predicted_label == 1 else "REAL"
    classification_label = "Likely Fake News" if predicted_label == 1 else "Likely Real News"
    model_confidence_percent = round(model_confidence * 100, 2)

    # Styling based on verdict
    if Category_Label == "REAL":
        colors = {"text_color": "#166534", "bg_color": "#f0fdf4", "accent_color": "#22c55e"}
        explanation = "The AI found no common patterns of misinformation."
    else:
        colors = {"text_color": "#b91c1c", "bg_color": "#fee2e2", "accent_color": "#ad2d2d"}
        explanation = "The AI detected deceptive linguistic patterns."

    # Highlight sentences for UI
    scored_sentences, parts = calculate_sentence_scores(text, lime_explanation or [])
    sentence_html = generate_sentence_highlight_html(parts, scored_sentences)

    return {
        "model_confidence": str(model_confidence_percent),
        "score_label": Category_Label,
        "classification_text": classification_label,
        "colors": colors,
        "supporting_articles": [], # Map from supporting_articles if needed
        "news_text": text,
        "extracted_claim": extracted_claim,
        "lime_explanation": lime_explanation, 
        "lime_html": sentence_html, 
        "influential_sentences": scored_sentences,
        "bias_data": bias_data,
        "explanation": explanation
    }