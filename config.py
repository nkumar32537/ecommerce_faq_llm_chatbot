# config.py

# Lightweight chat-tuned HF model for fallback generation
# You can swap this with a larger model (e.g., Llama 7B chat) if you have GPU memory.
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Cosine similarity threshold for when to trust the FAQ match
SIMILARITY_THRESHOLD = 0.80
