"""
Model Loader Module
Centralized model loading and management for CredibilityScan.
Refactored: Removed Deepfake, YOLO, and Sentiment pipelines.
Classification: English XGBoost + XLM-R Large Embeddings.
"""

import os
import torch
import spacy
import joblib
import numpy as np
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoModel

class ModelLoader:
    """Centralized model loader for the optimized Railway backend"""
    
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # English XGBoost + XLM-R Large Pipeline
        self.xgb_model = None
        self.nlp = None
        self.xlm_tokenizer = None
        self.xlm_model = None
        
        # Bias Detection Models (English Only)
        self.bias_vectorizer_en = None
        self.bias_model_en = None

        # Status tracking
        self.load_status = {}
    
    def load_all_models(self):
        """Load all text-based models and track their status"""
        print("\n" + "="*60 + "\nLOADING CENTRALIZED MODELS\n" + "="*60 + "\n")
        
        self.load_english_xgb_model()
        self.load_bias_models()
        
        print("\n" + "="*60 + "\nMODEL LOADING SUMMARY\n" + "="*60)
        for model_name, status in self.load_status.items():
            status_icon = "✓" if status else "✗"
            print(f"{status_icon} {model_name}")
        print("="*60 + "\n")

    def load_english_xgb_model(self):
        """Loads the XGBoost classifier and XLM-RoBERTa Large embedding components"""
        xgb_path = "Models/TF-IDF_English/credibility_scan_xgb_4096.pkl"
        roberta_name = "xlm-roberta-large"
        
        try:
            print("Loading English XGBoost Classifier & XLM-R Large Pipeline...")
            self.nlp = spacy.load("en_core_web_sm")
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
            self.xlm_model = AutoModel.from_pretrained(roberta_name, output_hidden_states=True)
            self.xlm_model.to(self.device)
            self.xlm_model.eval()
            self.xgb_model = joblib.load(xgb_path)
            
            print("✓ English XGBoost + XLM-R components loaded successfully!")
            self.load_status["English Classification (XGBoost + XLM-R)"] = True
        except Exception as e:
            print(f"✗ Error loading English XGBoost components: {e}")
            self.load_status["English Classification (XGBoost + XLM-R)"] = False

    def load_bias_models(self):
        """Loads English Bias Detection models"""
        base_path_en = "Models/biasenglish/" 
        
        try:
            print("Loading Bias Detection models (English)...")
            self.bias_vectorizer_en = joblib.load(os.path.join(base_path_en, "tfidf_vectorizer_english.pkl"))
            self.bias_model_en = joblib.load(os.path.join(base_path_en, "tfidf_bias_english_model.pkl"))
            print("✓ Bias detection models loaded successfully!")
            self.load_status["Bias Detection (English)"] = True
        except Exception as e:
            print(f"✗ Error loading Bias models: {e}")
            self.load_status["Bias Detection (English)"] = False

    def get_all_models(self) -> Dict:
        """Get all loaded models as a dictionary"""
        return {
            "xgb_model": self.xgb_model,
            "xlm_tokenizer": self.xlm_tokenizer,
            "xlm_model": self.xlm_model,
            "nlp": self.nlp,
            "bias_vectorizer_en": self.bias_vectorizer_en,
            "bias_model_en": self.bias_model_en
        }

# Singleton instance
_model_loader_instance = None

def get_model_loader():
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader()
        _model_loader_instance.load_all_models()
    return _model_loader_instance