# scripts/ingest2.py
"""
V2 ingest: uses the new HuggingFaceEmbeddings class.
Run once:
    python scripts/ingest2.py
"""

import bs4
import os
from pathlib import Path

# NEW import from langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()   # loads your .env into os.environ

# suppress the Windows symlink warning 
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# polite scraper header 
os.environ.setdefault("USER_AGENT", "ire-rag-llm/0.1 (github.com/adetuire)")

# index location 
INDEX_PATH = Path("data/faiss_index_v2.faiss")
INDEX_PATH.parent.mkdir(exist_ok=True)

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
store.save_local(str(INDEX_PATH))
print(f"[v2] FAISS index with {len(chunks)} chunks to {INDEX_PATH}")
