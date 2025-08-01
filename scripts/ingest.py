"""
Index Lilian Weng’s “LLM-Powered Autonomous Agents” blog post
into a FAISS vector store using a free HuggingFace embedder.
Run once:
    python scripts/ingest.py
"""

import bs4
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = Path("data/faiss_index.faiss")
INDEX_PATH.parent.mkdir(exist_ok=True)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()  

# split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and create FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store      = FAISS.from_documents(chunks, embeddings)

store.save_local(str(INDEX_PATH))
print(f"FAISS index with {len(chunks)} chunks saved to {INDEX_PATH}")