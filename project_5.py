import requests
from flask import Flask, request, jsonify
from rouge_score import rouge_scorer
import os

app = Flask(__name__)

# ✅ Replace with your actual Azure OpenAI API details
AZURE_GPT_API = "https://sakur-m8fg7upa-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
AZURE_API_KEY = "FciM7juJ1C9x8oDXTqrM6yLIC424RYguJCExpm0lSOBEBUs3g5UkJQQJ99BCACfhMk5XJ3w3AAAAACOGHVwt"  # 🚨 Don't expose API keys publicly!

# ✅ Function to analyze sentiment
def analyze_sentiment(text):
    headers = {
        "Authorization": f"Bearer {AZURE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Analyze the sentiment of the text."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.5
    }

    response = requests.post(AZURE_GPT_API, headers=headers, json=data)
    
    if response.status_code != 200:
        return {"error": "API request failed", "details": response.json()}

    response_data = response.json()
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return {"error": "Unexpected API response", "details": response_data}

# ✅ Function to summarize text
def summarize_text(text):
    headers = {
        "Authorization": f"Bearer {AZURE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Summarize the following text."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.5
    }

    response = requests.post(AZURE_GPT_API, headers=headers, json=data)

    if response.status_code != 200:
        return {"error": "API request failed", "details": response.json()}

    response_data = response.json()
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return {"error": "Unexpected API response", "details": response_data}

# ✅ Function to compute ROUGE score
def compute_rouge(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_summary, reference_summary)
    return scores

# ✅ Flask route for Sentiment Analysis
@app.route('/sentiment', methods=['POST'])
def sentiment_api():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment})

# ✅ Flask route for Summarization with ROUGE scoring
@app.route('/summarize', methods=['POST'])
def summarize_api():
    data = request.get_json()
    text = data.get("text", "").strip()
    reference_summary = data.get("reference_summary", "").strip()  # For ROUGE scoring

    if not text:
        return jsonify({"error": "No text provided"}), 400

    generated_summary = summarize_text(text)

    # Compute ROUGE score if reference summary is provided
    rouge_scores = None
    if reference_summary:
        rouge_scores = compute_rouge(generated_summary, reference_summary)

    return jsonify({
        "summary": generated_summary,
        "rouge_scores": rouge_scores if rouge_scores else "No reference summary provided"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Use Azure's assigned port, fallback to 5000 for local testing
    app.run(debug=True, host="0.0.0.0", port=port)

