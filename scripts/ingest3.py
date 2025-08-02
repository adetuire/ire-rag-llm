#!/usr/bin/env python3
"""
Build a FAISS index + pickled docstore for Lilian Weng’s agent blog post.

Usage:  python scripts/ingest3.py
"""

from __future__ import annotations
import faiss, pickle, os, sys
from pathlib import Path
from typing import List

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Configuration
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
]  # add more URLs here if you like

CHUNK_SIZE   = 1_000
CHUNK_OVERLAP = 200
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

FAISS_FP  = Path("data/faiss_index_v3.faiss")
PICKLE_FP = Path("data/store.pkl")
FAISS_FP.parent.mkdir(parents=True, exist_ok=True)

# Helpers
def _load_documents() -> List:
    """Download and return the blog post as a list[Document]."""
    strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    loader   = WebBaseLoader(web_paths=URLS, bs_kwargs=dict(parse_only=strainer))
    docs = loader.load()
    if not docs:
        raise RuntimeError("Document download failed – check your network.")
    return docs


def _split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def _build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    dim        = len(embeddings.embed_query("hello world"))
    index      = faiss.IndexFlatL2(dim)
    store      = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    store.add_documents(chunks)
    return store


# Main
def main() -> None:
    print("[ingest] downloading documents…")
    docs = _load_documents()
    print(f"[ingest] got {len(docs)=}")

    print("[ingest] splitting into chunks…")
    chunks = _split(docs)
    print(f"[ingest] produced {len(chunks)} chunks")

    print("[ingest] embedding + indexing…")
    vs = _build_vector_store(chunks)

    print(f"[ingest] saving FAISS index to {FAISS_FP}")
    vs.save_local(str(FAISS_FP))
    print(f"[ingest] pickling docstore to {PICKLE_FP}")
    with PICKLE_FP.open("wb") as f:
        pickle.dump(vs.docstore._dict, f)

    print("[ingest] done!!!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

