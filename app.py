from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import unicodedata
import re

# 1. DEFINE THE MODEL CLASSES (Same as your Colab)
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name):
        super(MultiTaskClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.sarcasm_head = nn.Linear(self.transformer.config.hidden_size, 2)
        self.humor_head = nn.Linear(self.transformer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.sarcasm_head(pooled_output), self.humor_head(pooled_output)

class SentimentClassifier(nn.Module):
    def __init__(self, model_name):
        super(SentimentClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# 2. SETUP APP & LOAD MODELS
app = FastAPI()

MODEL_NAME = "xlm-roberta-base"
device = torch.device("cpu") # Use CPU for free cloud hosting

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- NEW IMPORTS ---
from huggingface_hub import hf_hub_download

# ... (keep your class definitions MultiTaskClassifier etc. exactly as they were) ...

print("Loading Models...")
# Initialize model architecture
multi_task_model = MultiTaskClassifier(MODEL_NAME)
sentiment_model = SentimentClassifier(MODEL_NAME)

# --- NEW LOADING LOGIC ---
# Replace "YOUR_HF_USERNAME/sentiment-model-weights" with your ACTUAL Hugging Face Repo ID!
HF_REPO_ID = "YOUR_HF_USERNAME/sentiment-model-weights" 

try:
    print("Downloading weights from Hugging Face...")
    
    # Download the files directly from the cloud
    multi_task_path = hf_hub_download(repo_id=HF_REPO_ID, filename="multi_task_best_model.pt")
    sentiment_path = hf_hub_download(repo_id=HF_REPO_ID, filename="sentiment_best_model.pt")
    
    # Load them into the model
    multi_task_model.load_state_dict(torch.load(multi_task_path, map_location=device))
    sentiment_model.load_state_dict(torch.load(sentiment_path, map_location=device))
    print("Models loaded successfully!")

except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")

multi_task_model.to(device).eval()
sentiment_model.to(device).eval()

# 3. TEXT CLEANING FUNCTIONS
def clean_text(text):
    # (Simplified version of your cleaning logic for speed)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 4. API ENDPOINT
class AnalysisRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    text = clean_text(request.text)
    
    # Prepare inputs
    encoded = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=128, 
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    # Run Inference
    with torch.no_grad():
        sarcasm_logits, humor_logits = multi_task_model(input_ids, mask)
        sentiment_logits = sentiment_model(input_ids, mask)

        sarcasm_pred = torch.argmax(sarcasm_logits, dim=1).item()
        humor_pred = torch.argmax(humor_logits, dim=1).item()
        sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()

    # Map results
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
    return {
        "sarcasm": bool(sarcasm_pred == 1),
        "humor": bool(humor_pred == 1),
        "sentiment": sentiment_labels[sentiment_pred]
    }