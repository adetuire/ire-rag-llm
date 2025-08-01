# scripts/ingest.py
from typing import cast
import bs4, os, pickle
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

RAW_DIR = Path("data/raw")
VECTOR_FP = Path("data/faiss_index.faiss")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main() -> None:
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"[info] put source docs in {RAW_DIR}")
        return

    paths = list(RAW_DIR.glob("*"))
    if not paths:
        print(f"[warn] no documents found in {RAW_DIR}")
        return

    # load & chunk
    loader = WebBaseLoader(web_paths=[p.as_uri() for p in paths],
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer()))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 1/3 tagged "beginning", next third "middle", last third "end"
    total = len(chunks)
    third = total // 3

    from typing import Any, cast

    doc = cast(dict[str, Any], d.metadata)
    doc["section"] = "end"


    for i, d in enumerate(chunks):
        if i < third:
            d.metadata["section"] = "beginning"
        elif i < 2 * third:
            d.metadata["section"] = "middle"
        else:
            d.metadata["section"] = "end"

    # build in-memory store
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(VECTOR_FP))
    import pickle
    with open("data/store.pkl", "wb") as f:
        pickle.dump(store, f)

    print(f"built index with {len(chunks)} chunks → {VECTOR_FP}")
