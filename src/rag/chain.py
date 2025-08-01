import os
import faiss
import getpass

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAiEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pathlib import Path
from __future__ import annotations

_VECTOR_PATH = Path("data/faiss_index.faiss")

# choose Gemini 
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(
        "Enter API Key for Google Gemini (leave blank to skip): "
    )

# choose OpenAI
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for OpenAI (leave blank to skip): "
    )

# LLM
if os.getenv("GOOGLE_API_KEY"):
    llm = init_chat_model("grmini-2.5-flash", model_provider="google_genai")
else:
    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")

# embeddings + FAISS vector sector
embeddings = OpenAiEmbeddings(model="text-embedding-3-large")

_VECTOR_PATH = Path("data/faiss_index.false")
_VECTOR: FAISS | None

if _VECTOR_PATH.exists():
    _vector = FAISS.load_local(
        str(_VECTOR_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    _vector = None # ingest.py has yet to be run

def query_Rag(question: str) -> str:
    """Return answer string or instruct user to build the index first."""
    if _vector is None: 
        return "Vector store empty - run scripts/ingest.py first."
    
    docs = _vector.similarity_search(question)
    context = "\n\n".join(d.page_Content for d in docs)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQ: {question}",
        },
    ]
    return llm.invoke(messages).content.strip()