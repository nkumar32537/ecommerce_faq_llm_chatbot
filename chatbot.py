# chatbot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import Optional
from config import LLM_MODEL_NAME, SIMILARITY_THRESHOLD
from utils import load_faq_dataset, embed_questions, build_faiss_index
import numpy as np

# ---- Embedding & Index ----
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
DATASET_PATH = 'Ecommerce_FAQ_Chatbot_dataset.json'
faq_data = load_faq_dataset(DATASET_PATH)
faq_questions = [q['question'] for q in faq_data]

# Build index
faq_embeddings = embed_questions(faq_questions, embedding_model)
index = build_faiss_index(faq_embeddings)

# ---- LLM Fallback ----
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _detect_device()

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    llm_model.to(DEVICE)
except Exception as e:
    tokenizer = None
    llm_model = None
    print(f"[WARN] Could not load LLM model '{LLM_MODEL_NAME}': {e}")

def _generate_with_llm(prompt: str, max_new_tokens: int = 160) -> str:
    """Generate a response using the fallback LLM. If unavailable, return a safe message."""
    if tokenizer is None or llm_model is None:
        return ("I'm having trouble accessing the generative model right now. "
                "Please try rephrasing your question or ask about shipping, returns, orders, payments, etc.")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def get_faq_response(query: str) -> str:
    """Return best FAQ answer or fall back to LLM generation when similarity is low."""
    if not query or not query.strip():
        return "Please enter a question."
    query = query.strip()

    # Retrieve
    q_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_vec, k=1)
    top_score = float(D[0][0])
    top_idx = int(I[0][0])

    if top_score >= SIMILARITY_THRESHOLD:
        return f"**Q:** {faq_data[top_idx]['question']}\n\n**A:** {faq_data[top_idx]['answer']}"

    # Fallback to LLM
    prompt = f"""<|system|>
You are a helpful, accurate e-commerce assistant. Be concise and polite. If unsure, ask a short clarifying question.
</|system|>
<|user|>
{query}
</|user|>
<|assistant|>"""
    gen = _generate_with_llm(prompt)
    # Try to strip any prompt remnants
    if "</|assistant|>" in gen:
        gen = gen.split("</|assistant|>")[0]
    return f"**(Generated Answer)**\n\n{gen.strip()}"
