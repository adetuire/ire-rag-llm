"""
Create a FAISS index for Lilian Weng’s RAG blog post.
Run once:  python scripts/build_index.py
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()                     

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

URL = "https://lilianweng.github.io/posts/2023-06-15-agent/"
OUT_DIR = Path("data/faiss_blog")


def main() -> None:
    docs = WebBaseLoader(URL).load()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = FAISS.from_documents(docs, embeddings)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(OUT_DIR)
    print(f"Saved FAISS index to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
