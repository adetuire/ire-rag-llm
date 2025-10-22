from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.tools import tool

from rag.vector import vector_store as _vector_store


def _store():
    """Return the actual vector store instance (handles lru_cache factory or plain object)."""
    return _vector_store() if callable(_vector_store) else _vector_store


@dataclass
class RetrievedDocument:
    """Serializable representation of a retrieved document chunk."""

    document_id: Optional[str]
    chunk_id: Optional[str]
    title: Optional[str]
    preview: str
    score: Optional[float]
    metadata: Dict[str, Any]
    content: Optional[str] = None


def _make_preview(text: str, preview_chars: int) -> str:
    if len(text) <= preview_chars:
        return text
    truncated = text[:preview_chars].rsplit(" ", 1)[0]
    return f"{truncated.strip()}…"


def _similarity_with_scores(store, query: str, *, k: int, filters: Optional[Dict[str, Any]]):
    search_kwargs: Dict[str, Any] = {"k": k}
    if filters:
        search_kwargs["filter"] = filters

    if hasattr(store, "similarity_search_with_relevance_scores"):
        return store.similarity_search_with_relevance_scores(query, **search_kwargs)
    if hasattr(store, "similarity_search_with_score"):
        return store.similarity_search_with_score(query, **search_kwargs)

    docs = store.similarity_search(query, **search_kwargs)
    return [(doc, None) for doc in docs]


def _serialize(documents: Iterable[RetrievedDocument]) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        parts.append(
            "\n".join(
                [
                    f"Result {idx}",
                    f"Document ID: {doc.document_id}",
                    f"Chunk ID: {doc.chunk_id}",
                    f"Title: {doc.title}",
                    f"Score: {doc.score}",
                    f"Preview: {doc.preview}",
                    f"Metadata: {doc.metadata}",
                ]
            )
        )
    return "\n\n".join(parts)


def _retrieve_impl(
    query: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 4,
    preview_chars: int = 280,
    include_content: bool = False,
) -> Tuple[str, List[RetrievedDocument]]:
    store = _store()
    results_with_scores = _similarity_with_scores(store, query, k=limit, filters=filters)

    structured: List[RetrievedDocument] = []
    for doc, score in results_with_scores:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        page_content = getattr(doc, "page_content", "")
        structured.append(
            RetrievedDocument(
                document_id=metadata.get("document_id") or metadata.get("source"),
                chunk_id=metadata.get("chunk_id"),
                title=metadata.get("title") or metadata.get("document_title"),
                preview=_make_preview(page_content, preview_chars),
                score=float(score) if score is not None else None,
                metadata=metadata,
                content=page_content if include_content else None,
            )
        )

    serialized = _serialize(structured)
    return serialized, structured


@tool(response_format="content_and_artifact")
def retrieve(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 4,
    preview_chars: int = 280,
):
    """Retrieve information related to a query with optional metadata filters."""

    return _retrieve_impl(query, filters=filters, limit=limit, preview_chars=preview_chars)


def retrieve_raw(
    query: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 4,
    preview_chars: int = 280,
    include_content: bool = True,
):
    """Programmatic access to structured retrieval results."""

    return _retrieve_impl(
        query,
        filters=filters,
        limit=limit,
        preview_chars=preview_chars,
        include_content=include_content,
    )
