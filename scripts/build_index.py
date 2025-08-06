"""
Build a FAISS index from a folder of .txt files (offline).

Run:
  python scripts/build_index.py --docs data/wiki_docs --out data/faiss_wiki
  # optional:
  # --model sentence-transformers/all-MiniLM-L6-v
"""
import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def load_docs(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    paths = list(docs_dir.rglob("*.txt"))
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")
            continue
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Folder containing .txt files")
    ap.add_argument("--out", required=True, help="Output folder for FAISS index")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model name",
    )
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    out_dir = Path(args.out)
    if not docs_dir.exists():
        print(f"[error] docs dir not found: {docs_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(docs_dir)
    print(f"Loaded {len(docs)} documents from {docs_dir}")
    if not docs:
        print("[error] No .txt files found. Did the extractor write to this folder?")
        return 1

    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(out_dir)

    print(f"Saved FAISS index to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
