from __future__ import annotations
import argparse, os, getpass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

INDEX_PATH = Path("data/faiss_index.faiss")
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Index not found at {INDEX_PATH}. Run `python scripts/ingest2.py` first.")

llm = None

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector     = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    k: int
    context: list[Document]
    answer: str

def retrieve(state: State) -> dict[str, list[Document]]:
    # pull out the number of chunks to fetch
    top_k = state.get("k", 4)
    retrieved_docs = vector.similarity_search(state["question"], k=top_k)
    return {"context": retrieved_docs}

# pull the RAG prompt from LangChain’s hub
prompt = hub.pull("rlm/rag-prompt")

def generate(state: State) -> dict[str, str]:
    docs_content = "\n\n".join(d.page_content for d in state["context"])
    messages = prompt.invoke({
      "question": state["question"],
      "context": docs_content
    }).to_messages()
    response = llm.invoke(messages)
    return {"answer": response.content}

graph = (
  StateGraph(State)
  .add_sequence([retrieve, generate])
  .add_edge(START, "retrieve")
  .compile()
)

