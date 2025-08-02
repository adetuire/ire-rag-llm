from functools import lru_cache
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


@lru_cache(maxsize=1)
def vector_store():
    """
    Return a singleton FAISS retriever for the blog-post vectors.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    faiss_dir = repo_root / "data" / "faiss_blog"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True,            # LC ≥0.3 safeguard
    )
    return store.as_retriever(search_kwargs={"k": 2})
