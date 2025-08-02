# scripts/ingest2.py
"""
A minimal one-shot ingester that downloads Lilian Weng’s agent post,
chunks it and builds a FAISS v2 index.

Run once:
    python scripts/ingest2.py
"""

import bs4
import os
from pathlib import Path

# NEW import from langchain_huggingface
import bs4
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
load_dotenv()   # loads your .env into os.environ

# suppress the Windows symlink warning 
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# polite scraper header 
os.environ.setdefault("USER_AGENT", "ire-rag-llm/0.1 (github.com/adetuire)")

# index location 
INDEX_FP = Path("data/faiss_index_v2.faiss")
INDEX_FP.parent.mkdir(exist_ok=True)

# Download & parse the blog post
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

#Split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks   = splitter.split_documents(docs)

# Embed with the new HF embedder
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store      = FAISS.from_documents(chunks, embeddings)

# Persist
store.save_local(str(INDEX_FP))
print(f"[v2] FAISS index with {len(chunks)} chunks to {INDEX_FP}")
