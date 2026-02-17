"""
Model Loader Module
Centralized model loading and management for CredibilityScan
Refactored: Removed Tagalog pipelines, Upgraded English to XGBoost + XLM-R Large
"""

import os
import torch
import spacy
import joblib
import numpy as np
from typing import Dict, List, Union
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class ModelLoader:
    """Centralized model loader for all ML models used in the application"""
    
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Deepfake Detection
        self.deepfake_model = None
        self.yolo_model = None
        
        # English XGBoost + XLM-R Large Pipeline
        self.xgb_model = None
        self.nlp = None
        self.xlm_tokenizer = None
        self.xlm_model = None
        
        # Shared Sentiment Model
        self.sentiment_pipeline = None
        
        # Bias Detection Models (English Only)
        self.bias_vectorizer_en = None
        self.bias_model_en = None

        # Status tracking
        self.load_status = {}
    
    def load_all_models(self):
        """Load all models and track their status"""
        print("\n" + "="*60 + "\nLOADING ALL MODELS\n" + "="*60 + "\n")
        
        self.load_deepfake_model()
        self.load_yolo_model()
        self.load_english_xgb_model()
        self.load_sentiment_model()
        self.load_bias_models()
        
        print("\n" + "="*60 + "\nMODEL LOADING SUMMARY\n" + "="*60)
        for model_name, status in self.load_status.items():
            status_icon = "✓" if status else "✗"
            print(f"{status_icon} {model_name}")
        print("="*60 + "\n")
    
    def load_sentiment_model(self):
        """Loads the Multilingual Sentiment Model (XLM-RoBERTa Base)"""
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        try:
            print(f"Loading Sentiment Model ({model_name})...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Sentiment pipeline loaded successfully!")
            self.load_status["Sentiment Analysis (Multilingual)"] = True
        except Exception as e:
            print(f"✗ Error loading Sentiment model: {e}")
            self.load_status["Sentiment Analysis (Multilingual)"] = False


    def load_english_xgb_model(self):
        """Loads the XGBoost classifier and XLM-RoBERTa Large embedding components"""
        xgb_path = "C:/git/FakeNewsApp-with-LimeTagalog-/Models/TF-IDF_English/credibility_scan_xgb_4096.pkl"
        roberta_name = "xlm-roberta-large"
        
        try:
            print("Loading English XGBoost Classifier & XLM-R Large Pipeline...")
            
            # 1. Spacy
            self.nlp = spacy.load("en_core_web_sm")
            
            # 2. XLM-R Large
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
            self.xlm_model = AutoModel.from_pretrained(roberta_name, output_hidden_states=True)
            self.xlm_model.to(self.device)
            self.xlm_model.eval()
            
            # 3. XGBoost
            self.xgb_model = joblib.load(xgb_path)
            
            print("✓ English XGBoost + XLM-R components loaded successfully!")
            self.load_status["English Classification (XGBoost + XLM-R)"] = True
        except Exception as e:
            print(f"✗ Error loading English XGBoost components: {e}")
            self.load_status["English Classification (XGBoost + XLM-R)"] = False

    def load_bias_models(self):
        """Loads English Bias Detection models"""
        base_path_en = "C:/git/FakeNewsApp-with-LimeTagalog-/Models/biasenglish/" 
        
        try:
            print("Loading Bias Detection models (English)...")
            self.bias_vectorizer_en = joblib.load(os.path.join(base_path_en, "tfidf_vectorizer_english.pkl"))
            self.bias_model_en = joblib.load(os.path.join(base_path_en, "tfidf_bias_english_model.pkl"))
            
            print("✓ Bias detection models loaded successfully!")
            self.load_status["Bias Detection (English)"] = True
        except Exception as e:
            print(f"✗ Error loading Bias models: {e}")
            self.load_status["Bias Detection (English)"] = False

    # --- Embedding Helper Methods ---

    def embed_sentences_concat_large(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Tokenize batches, pass through XLM-R Large, concatenate last 4 layers, 
        and apply mean pooling.
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            
            encoded_input = self.xlm_tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=256, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.xlm_model(**encoded_input)
                hidden_states = outputs.hidden_states
                
                # Concatenate last 4 layers [-1, -2, -3, -4] on dim=-1
                last_four_layers = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)
                
                # Attention-mask-based mean pooling
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_four_layers.size()).float()
                
                sum_embeddings = torch.sum(last_four_layers * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                batch_embeddings = sum_embeddings / sum_mask
                all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def article_to_embedding(self, article: str) -> np.ndarray:
        """
        Splits article into sentences using Spacy, filters short ones,
        and returns the mean embedding of all sentences (4096-D).
        """
        if not article or not article.strip():
            return np.zeros(4096)

        doc = self.nlp(article)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

        if not sentences:
            return np.zeros(4096)

        sentence_embeddings = self.embed_sentences_concat_large(sentences)
        return np.mean(sentence_embeddings, axis=0)

    def get_all_models(self) -> Dict:
        """Get all loaded models as a dictionary"""
        return {
            "deepfake_model": self.deepfake_model,
            "yolo_model": self.yolo_model,
            "xgb_model": self.xgb_model,
            "xlm_tokenizer": self.xlm_tokenizer,
            "xlm_model": self.xlm_model,
            "nlp": self.nlp,
            "sentiment_pipeline": self.sentiment_pipeline,
            "bias_vectorizer_en": self.bias_vectorizer_en,
            "bias_model_en": self.bias_model_en
        }

_model_loader_instance = None
def get_model_loader() -> ModelLoader:
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader()
        _model_loader_instance.load_all_models()
    return _model_loader_instance
