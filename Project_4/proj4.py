import openai
import os
import logging
import time

# Configure logging
logging.basicConfig(filename="chatbot_logs.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API credentials
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://sakur-m8fg7upa-swedencentral.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "FciM7juJ1C9x8oDXTqrM6yLIC424RYguJCExpm0lSOBEBUs3g5UkJQQJ99BCACfhMk5XJ3w3AAAAACOGHVwt")
API_VERSION = "2024-05-01-preview"  

# Validate API Key
if not API_KEY:
    raise ValueError("Error: Missing API Key. Set AZURE_OPENAI_API_KEY as an environment variable.")

# Configure OpenAI API for Azure
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = API_KEY
openai.api_version = API_VERSION



def generate_response(prompt):
    """Generates a response from Azure OpenAI with monitoring and logging."""
    start_time = time.time()  # Start measuring response time
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        end_time = time.time()  # Stop measuring response time
        response_time = end_time - start_time

        # Log user input and response time
        logging.info(f"Prompt: {prompt} | Response Time: {response_time:.2f} seconds")

        # Store responses for analysis
        with open("responses_log.txt", "a") as f:
            f.write(f"User: {prompt}\nBot: {response['choices'][0]['message']['content']}\n\n")

        return response["choices"][0]["message"]["content"].strip()
    
    except openai.error.InvalidRequestError as e:
        logging.error(f"Invalid Request Error: {e}")
        return f"Error: {e}"
    except openai.error.AuthenticationError:
        logging.error("Authentication Error: Invalid API key or incorrect endpoint.")
        return "Error: Invalid API key or incorrect endpoint."
    except Exception as e:
        logging.error(f"General Error: {e}")
        return f"Error: {e}"

def evaluate_performance():
    """Analyzes chatbot logs and provides average response time, error count, and performance metrics."""
    total_time = 0
    count = 0
    error_count = 0
    
    with open("chatbot_logs.txt", "r") as log_file:
        for line in log_file:
            if "Response Time" in line:
                time_taken = float(line.split("Response Time: ")[1].split(" seconds")[0])
                total_time += time_taken
                count += 1
            if "Error" in line:
                error_count += 1

    if count > 0:
        avg_time = total_time / count
        print(f"📊 Average Response Time: {avg_time:.2f} seconds")
        print(f"⚠️ Total Errors Encountered: {error_count}")
    else:
        print("No valid response times recorded.")


# List of predefined questions
questions = [
    "Tell me a joke!"
]

# Run predefined questions
print("\n=== Running Predefined Questions ===\n")
for question in questions:
    print(f"You: {question}")
    bot_response = generate_response(question)
    print(f"Bot: {bot_response}\n")

# Call the evaluation function at the end
evaluate_performance()
