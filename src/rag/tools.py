# src/rag/tools.py
from __future__ import annotations

import os
from langchain_core.tools import tool

# IMPORTANT: import the factory and CALL IT to get the store instance
# If in your repo it's named get_vector_store(), adjust the import and call.
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


# Helper used by your conversational chain when you need (content, docs) directly
def retrieve_raw(query: str):
    return _retrieve_impl(query)
