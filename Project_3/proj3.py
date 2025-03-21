import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# ✅ Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Load the fine-tuned model
model_path = "fine_tuned_bert"  # Ensure this matches your saved model folder
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    print("✅ Fine-tuned model loaded successfully!")
except:
    print("⚠️ Failed to load fine-tuned model! Ensure the model is correctly saved.")

# ✅ Create a QA Pipeline with Increased max_length
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# ✅ Define test examples (Increased context size)
test_examples = [
    {
        "question": "What is the tallest mountain in the world?",
        "context": "Mount Everest, located in the Himalayas, is the tallest mountain in the world, reaching an elevation of 8,848 meters. It is the highest point on Earth."
    },
    {
        "question": "Who discovered gravity?",
        "context": "Gravity was first described by Sir Isaac Newton when he formulated the laws of motion and universal gravitation in the 17th century."
    },
    {
        "question": "What is the chemical formula of water?",
        "context": "Water is a compound made up of two hydrogen atoms and one oxygen atom, represented by the chemical formula H2O."
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "context": "'Romeo and Juliet' is a famous play written by William Shakespeare in the late 16th century."
    },
    {
        "question": "What is the capital of Japan?",
        "context": "Tokyo is the capital of Japan, known for its modern skyscrapers, historic temples, and bustling markets."
    },
]

# ✅ Run inference with increased max_length and handle truncation
print("\n===== QA Pipeline Predictions (Updated) =====")
for i, example in enumerate(test_examples, 1):
    output = qa_pipeline({
        "question": example["question"],
        "context": example["context"]
    })
    print(f"\nExample {i}:")
    print(f"Question: {example['question']}")
    print(f"Predicted Answer: {output['answer']} (Confidence: {output['score']:.4f})\n")

# ✅ Debugging: Check tokenization for one of the failed cases
example = test_examples[0]  # Change index if needed
tokens = tokenizer(
    example["question"], 
    example["context"], 
    return_offsets_mapping=True, 
    truncation=True, 
    max_length=512,  # Increased max_length
    padding="max_length"
)

print("\n===== Tokenization Debugging (Updated) =====")
print("Tokenized Example IDs:")
print(tokens["input_ids"][:50])  # Show first 50 token IDs
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
print("\nDecoded Tokens (first 100 tokens):")
print(decoded_tokens[:100])  # Print first 100 tokens for clarity

# ✅ Ensure the answer exists in the tokenized input
if "Mount Everest" not in decoded_tokens:
    print("⚠️ Answer is still missing in tokenized input! Consider increasing max_length further.")

# ✅ Manually Extract Answers Using Model Logits
print("\n===== Manual Answer Extraction (Updated) =====")
tokens = tokenizer(
    example["question"], 
    example["context"], 
    truncation=True, 
    max_length=512, 
    padding="max_length",
    return_tensors="pt"
)

# Move tensors to GPU if available
tokens = {k: v.to(device) for k, v in tokens.items()}

# Get model outputs
with torch.no_grad():
    outputs = model(**tokens)

# Extract start and end positions
start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores) + 1  # End index is exclusive

# Decode answer
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(tokens["input_ids"][0][start_index:end_index])
)

print(f"Predicted Answer (Manual Extraction): {answer}")

# ✅ If issues persist, try a better pre-trained model
use_alternate_model = False  # Set to True to use RoBERTa instead of fine-tuned BERT

if use_alternate_model:
    print("\n===== Trying a Larger Model: RoBERTa =====")
    model_checkpoint = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)

    # Create new QA pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Run inference again with RoBERTa
    for i, example in enumerate(test_examples, 1):
        output = qa_pipeline(example)
        print(f"\nExample {i} (RoBERTa):")
        print(f"Question: {example['question']}")
        print(f"Predicted Answer: {output['answer']} (Confidence: {output['score']:.4f})\n")

print("\n✅ Debugging Complete! Check outputs above.")










