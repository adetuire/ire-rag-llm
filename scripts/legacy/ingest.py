# scripts/ingest.py
# scripts/ingest.py
"""
Index any local .html / .md / .txt files you drop in "data/raw/".
Usage:  python scripts/ingest.py
Creates:
    • data/faiss_index.faiss
    • data/store.pkl             
"""

from bs4.filter import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import pickle

from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

VEC_FP = Path("data/faiss_index.faiss")
PICKLE_FP = Path("data/store.pkl")

EMBEDDER = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def _load_documents() -> List:
    paths = list(RAW_DIR.glob("*"))
    if not paths:
        raise RuntimeError(
            f"No documents found in {RAW_DIR}. "
            "Drop .html/.md/.txt files there first."
        )

    loader = WebBaseLoader(
        web_paths=[p.as_uri() for p in paths],
        bs_kwargs=dict(parse_only=SoupStrainer()),
    )
    return loader.load()


def main() -> None:
    docs = _load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # tag thirds as beginning / middle / end (useful for demo query-analysis)
    third = len(chunks) // 3
    for i, d in enumerate(chunks):
        if i < third:
            d.metadata["section"] = "beginning"
        elif i < 2 * third:
            d.metadata["section"] = "middle"
        else:
            d.metadata["section"] = "end"

    store = FAISS.from_documents(chunks, EMBEDDER)

    VEC_FP.parent.mkdir(exist_ok=True)
    store.save_local(str(VEC_FP))
    with PICKLE_FP.open("wb") as f:
        pickle.dump(store, f)

    print(f"[ingest] indexed {len(chunks)} chunks → {VEC_FP}")


if __name__ == "__main__":
    main()