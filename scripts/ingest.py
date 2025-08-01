# scripts/ingest.py
"""
Downloads Lilian Weng’s “LLM-Powered Autonomous Agents” blog post
Splits into ~1 000-char chunks
Embeds with text-embedding-3-large
Saves a single FAISS index to data/faiss_index.faiss
Run:
    python scripts/ingest.py
"""
import bs4, os, pickle
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

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
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
store = FAISS.from_documents(chunks, embeddings)

store.save_local(str(INDEX_PATH))
print(f"FAISS index with {len(chunks)} chunks saved to {INDEX_PATH}")