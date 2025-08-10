# utils.py

import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

def load_faq_dataset(path: str) -> List[dict]:
    """Load the FAQ dataset in the provided JSON format."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError("Invalid dataset format: expected a dict with 'questions' key.")

def embed_questions(questions: List[str], model: SentenceTransformer) -> np.ndarray:
    """Create normalized embeddings for a list of questions."""
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings: np.ndarray):
    """Build an inner-product FAISS index (works as cosine similarity with normalized vectors)."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index
