import pandas as pd
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from flask import Flask, request, jsonify

# Step 1: Check for GPU and Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Parse command-line arguments for dataset path (default to 'reviews.csv')
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="reviews.csv", help="Path to dataset CSV file")
args = parser.parse_args()

# Step 3: Load and Prepare Dataset
df = pd.read_csv(args.dataset_path)  # Load dataset from user-specified path

# Preprocess data
df = df.sample(n=1000, random_state=42)  # Sample 1000 for quick training
df.rename(columns={'content': 'review', 'score': 'sentiment'}, inplace=True)
df.dropna(subset=['review', 'sentiment'], inplace=True)

# Convert sentiment scores to categories
def map_score_to_sentiment(score):
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment_category'] = df['sentiment'].apply(map_score_to_sentiment)

# Step 4: Split Data for Training
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].tolist(),
    df['sentiment_category'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist(),
    test_size=0.2,
    random_state=42
)

# Step 5: Tokenization using Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding="max_length", max_length=512)

# Step 6: Create PyTorch Dataset Class with Fix
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}  # Convert dataset to dictionary
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}  # Properly index tensors
        item["labels"] = self.labels[idx]
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Step 7: Load Pretrained Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)  # Move model to GPU if available

# Step 8: Define Training Arguments with Fixes
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # Fixed deprecated warning
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision if using CUDA
    dataloader_pin_memory=True  # Fix for CUDA memory pinning issue
)

# Step 9: Define Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Step 10: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Step 11: Save the Trained Model
model_path = "sentiment_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")

# Step 12: Create a Cloud-Based API Using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400

        text = data["text"]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).cpu().item()  # Ensure it's on CPU
        
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        response = {"sentiment": sentiment_map[pred_label]}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Sentiment Analysis API...")
    app.run(host='0.0.0.0', port=5000)
