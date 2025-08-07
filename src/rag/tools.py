from __future__ import annotations

from langchain_core.tools import tool
from rag.vector import vector_store as _vector_store

def _store():
    """Return the actual vector store instance (handles lru_cache factory or plain object)."""
    return _vector_store() if callable(_vector_store) else _vector_store

def _retrieve_impl(query: str):
    store = _store()
    retriever = store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    serialized = "\n\n".join(
        f"Source: {getattr(d, 'metadata', {})}\nContent: {getattr(d, 'page_content', '')}"
        for d in docs
    )
    return serialized, docs

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    return _retrieve_impl(query)

def retrieve_raw(query: str):
    return _retrieve_impl(query)
