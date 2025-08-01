# src/rag/chain2.py
"""
V2 chain: loads `data/faiss_index_v2.faiss` with the new HF embedder.
Pure retrieval, no paid API calls.
"""

from __future__ import annotations
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# disable symlink warnings on Windows 
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

#  scraper header (matches ingest2) 
os.environ.setdefault("USER_AGENT", "ire-rag-llm/0.1 (github.com/adetuire)")

INDEX_PATH = Path("data/faiss_index_v2.faiss")
if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"[v2] index not found at {INDEX_PATH} — run scripts/ingest2.py first."
    )

# Use the *same* new embedder class
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector     = FAISS.load_local(str(INDEX_PATH), embeddings)

def retrieve_v2(question: str, k: int = 4) -> list[str]:
    """
    Return the top-k document chunks most similar to question.
    """
    docs = vector.similarity_search(question, k=k)
    return [d.page_content for d in docs]
