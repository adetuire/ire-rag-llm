# src/rag/tools.py
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve top-2 docs relevant to `query`."""
    from .vector import vector_store          # reuse from Part 1
    docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {d.metadata}\nContent: {d.page_content}"
        for d in docs
    )
    return serialized, docs

@tool(response_format="content_and_artifact")
def retrieve_v2(query: str, k: int = 4):
    """Retrieve top-k docs relevant to `query`."""
    from .chain2 import retrieve_v2          # reuse from Part 2
    docs = retrieve_v2(query, k)
    serialized = "\n\n".join(
        f"Content: {d}" for d in docs
    )
    return serialized, docs