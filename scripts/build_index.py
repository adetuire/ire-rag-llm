"""
Build a FAISS index from a folder of .txt files (offline).
Run:
  python scripts/build_index.py --docs data/wiki_docs --out data/faiss_wiki
"""
from pathlib import Path
#from dotenv import load_dotenv

#load_dotenv()                     

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def load_docs(docs_dir: Path) -> list[Document]:
    docs = []
    for p in docs_dir.rglob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Folder of .txt files")
    ap.add_argument("--out", required=True, help="Output folder for FAISS index")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(docs_dir)
    print(f"Loaded {len(docs)} docs from {docs_dir}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(out_dir)

    print(f"✅ Saved FAISS index to {out_dir}")

if __name__ == "__main__":
    main()
