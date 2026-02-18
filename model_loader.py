"""
Model Loader Module
Centralized model loading and management for CredibilityScan.
Matches training logic: XLM-R Large (Last 4 Layers Concatenated) -> 4096 Dimensions.
"""

import os
import torch
import spacy
import joblib
import numpy as np
from typing import Dict
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
        # Ensure this matches your file path exactly
        xgb_path = "Models/TF-IDF_English/credibility_scan_xgb_4096.pkl"
        roberta_name = "xlm-roberta-large"
        
        try:
            print("Loading English XGBoost Classifier & XLM-R Large Pipeline...")
            
            # Load Spacy for sentence splitting (Crucial for your logic)
            if not spacy.util.is_package("en_core_web_sm"):
                spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

            # Load Tokenizer & Model
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
            # IMPORTANT: output_hidden_states=True is required for your 4096 logic
            self.xlm_model = AutoModel.from_pretrained(roberta_name, output_hidden_states=True)
            self.xlm_model.to(self.device)
            self.xlm_model.eval()
            
            # Load XGBoost
            self.xgb_model = joblib.load(xgb_path)
            
            print("✓ English XGBoost + XLM-R (4096-dim) loaded successfully!")
            self.load_status["English Classification"] = True
        except Exception as e:
            print(f"✗ Error loading English XGBoost components: {e}")
            self.load_status["English Classification"] = False

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

    def article_to_embedding(self, text: str):
        """
        MATCHES TRAINING NOTEBOOK LOGIC:
        1. Split article into sentences.
        2. Embed each sentence by concatenating Last 4 Hidden Layers of XLM-R.
        3. Mean pool the result to get 4096 dimensions.
        """
        if not text or not self.xlm_tokenizer or not self.xlm_model:
            print("Error: Text is empty or XLM-R model is not loaded.")
            return np.zeros(4096) # Fail safe return

        try:
            # 1. Split into sentences using Spacy
            doc = self.nlp(text[:100000]) # Truncate massive text to prevent OOM
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            
            if len(sentences) == 0:
                # If no valid sentences, assume input is just the raw text
                sentences = [text]

            # Process in batches to avoid OOM on Railway
            batch_size = 8
            all_embeddings = []

            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Tokenize
                inputs = self.xlm_tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=256, # Matches your notebook
                    return_tensors="pt"
                ).to(self.device)

                # Inference
                with torch.no_grad():
                    outputs = self.xlm_model(**inputs)

                # --- 4096 DIMENSION LOGIC ---
                # Concatenate last 4 hidden layers (1024 * 4 = 4096)
                hidden_states = outputs.hidden_states
                last_four = [hidden_states[i] for i in [-1, -2, -3, -4]]
                concatenated = torch.cat(last_four, dim=-1)
                
                # Mean Pooling (Attention Mask applied)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(concatenated.size()).float()
                summed = torch.sum(concatenated * mask, 1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                
                # Result is (Batch_Size, 4096)
                batch_embeddings = (summed / counts).cpu().numpy()
                all_embeddings.append(batch_embeddings)

            # Stack all batches and take the mean across all sentences
            # Final shape: (4096,)
            final_embedding = np.vstack(all_embeddings).mean(axis=0)
            
            return final_embedding

        except Exception as e:
            print(f"Error in article_to_embedding: {e}")
            return np.zeros(4096) # Return zero vector on failure

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