from langchain_core.tools import tool
from src.rag.vector import vector_store


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve the top-2 chunks from Lilian Weng’s blog post that
    best match query.
    """
    docs = vector_store().similarity_search(query, k=2)
    text = "\n\n".join(
        f"Source: {d.metadata.get('source', 'blog')}\nContent: {d.page_content}"
        for d in docs
    )
    return text, docs
