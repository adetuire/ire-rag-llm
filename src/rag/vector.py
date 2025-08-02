from functools import lru_cache
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

@lru_cache(maxsize=1)
def vector_store():
    """Return a singleton FAISS retriever (blog post)."""
    dir_ = Path(__file__).parent.parent.parent / "data/faiss_blog"
    return FAISS.load_local(
        dir_,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True,
    ).as_retriever(search_kwargs={"k": 2})
