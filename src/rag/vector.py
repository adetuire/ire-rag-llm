from functools import lru_cache
from pathlib import Path
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def vector_store():
    """
    Return the FAISS index selected by RAG_INDEX_DIR (env).
    Falls back to the blog index for backwards compatibility.
    """
    repo_root = Path(__file__).resolve().parents[2]
    index_dir = os.getenv("RAG_INDEX_DIR")
    if index_dir:
        faiss_dir = Path(index_dir)
    else:
        faiss_dir = repo_root / "data" / "faiss_blog"   # legacy default

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    return vs  # callers use .similarity_search(...)