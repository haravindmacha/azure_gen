import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Replace with your actual Azure OpenAI API details
AZURE_GPT_API = "https://sakur-m8fg7upa-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
AZURE_API_KEY = "FciM7juJ1C9x8oDXTqrM6yLIC424RYguJCExpm0lSOBEBUs3g5UkJQQJ99BCACfhMk5XJ3w3AAAAACOGHVwt"

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
    
    # ✅ Improved Error Handling
    if response.status_code != 200:
        return {"error": "API request failed", "details": response.json()}

    response_data = response.json()

    # ✅ Ensure 'choices' exists in response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return {"error": "Unexpected API response", "details": response_data}

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

    # ✅ Improved Error Handling
    if response.status_code != 200:
        return {"error": "API request failed", "details": response.json()}

    response_data = response.json()

    # ✅ Ensure 'choices' exists in response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return {"error": "Unexpected API response", "details": response_data}

@app.route('/sentiment', methods=['POST'])
def sentiment_api():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment})

@app.route('/summarize', methods=['POST'])
def summarize_api():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarize_text(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # ✅ Allows external access
