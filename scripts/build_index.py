"""
Run once to create vectors for Lilian Weng's RAG blog post.
  python scripts/build_index.py
"""

from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

URL = "https://lilianweng.github.io/posts/2023-06-15-agent/"
OUT_DIR = Path("data/faiss_blog")           # git-ignored via .gitignore

docs = WebBaseLoader(URL).load()
emb = OpenAIEmbeddings(model="text-embedding-3-small")
faiss = FAISS.from_documents(docs, emb)
faiss.save_local(OUT_DIR)
print("Saved FAISS index to", OUT_DIR)
