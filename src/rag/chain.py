from __future__ import annotations

import os
import faiss
import getpass

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAiEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 
_VECTOR_PATH = Path("data/faiss_index.faiss")

# choose Gemini 
if not os.getenv("GOOGLE_API_KEY"):
    key = getpass.getpass("Enter API Key for Google Gemini (leave blank to skip): ")
    if key:
        os.environ["GOOGLE_API_KEY"] = key 

# choose OpenAI
if not os.getenv("OPENAI_API_KEY"):
    key = getpass.getpass("Enter API key for OpenAI (leave blank to skip): ")
    if key:
        os.environ["OPENAI_API_KEY"] = key 

# LLM
if os.getenv("GOOGLE_API_KEY"):
    llm = init_chat_model("grmini-2.5-flash", model_provider="google_genai")
else:
    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")

VECTOR_PATH = Path("data/faiss_index.false")

# embeddings + FAISS vector sector
embeddings = OpenAiEmbeddings(model="text-embedding-3-large")

if not VECTOR_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found at {VECTOR_PATH}. "
        "Run  python scripts/ingest.py  first."
    )

vector = FAISS.load_local(
    str(VECTOR_PATH),
    embeddings,
    allow_dangerous_deserialization=True,   # needed when saving with pickle
)

def query_rag(question: str) -> str:
    """
    Retrieve-and-generate answer for *question* using the pre-built FAISS store.
    """
    docs = vector.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user",   "content": f"Context:\n{context}\n\nQ: {question}"},
    ]
    return llm.invoke(messages).content.strip()
