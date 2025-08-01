"""
Pure retrieval: load the FAISS index and return top-k chunks for any question.
No API keys, no paid calls. Free, local embedding + FAISS only.
"""

from __future__ import annotations
import os
import faiss
import getpass

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import argparse

# choose Gemini 
#if not os.getenv("GOOGLE_API_KEY"):
#    key = getpass.getpass("Enter API Key for Google Gemini (leave blank to skip): ")
#    if key:
#        os.environ["GOOGLE_API_KEY"] = key 

# choose OpenAI
#if not os.getenv("OPENAI_API_KEY"):
#    key = getpass.getpass("Enter API key for OpenAI (leave blank to skip): ")
#    if key:
#        os.environ["OPENAI_API_KEY"] = key 

# LLM
#if os.getenv("GOOGLE_API_KEY"):
#    llm = init_chat_model("grmini-2.5-flash", model_provider="google_genai")
#else:
#    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")

INDEX_PATH = Path("data/faiss_index.faiss")
if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Index not found at {INDEX_PATH}. Run `python scripts/ingest.py` first."
    )

# embeddings + FAISS vector sector
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True,   # safe for your own index
)


def retrieve(question: str, k: int = 4) -> list[str]:
    """
    Return the top-k document chunks most similar to *question*.
    """
    docs = vector.similarity_search(question, k=k)
    return [d.page_content for d in docs]

def main():
    p = argparse.ArgumentParser(
        prog="python -m src.rag.chain",
        description="Run a quick similarity search against your FAISS index."
    )
    p.add_argument("--question", "-q", required=True, help="Your query string")
    p.add_argument("--k", type=int, default=4, help="Number of chunks to return")
    args = p.parse_args()
    print(retrieve(args.question, k=args.k))

if __name__ == "__main__":
    main()