# src/rag/chain_rag.py
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Literal, List, TypedDict

from typing_extensions import Annotated
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_FP = Path("data/faiss_index.faiss")
PICKLE_FP = Path("data/store.pkl")

if PICKLE_FP.exists():
    with PICKLE_FP.open("rb") as f:
        vector_store = pickle.load(f)
    print("[chain_rag] loaded in-memory vector store (pickle)")
elif FAISS_FP.exists():
    vector_store = FAISS.load_local(
        str(FAISS_FP),
        HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )
    print("[chain_rag] loaded FAISS index from disk")
else:
    raise FileNotFoundError(
        "No vector store found. Run scripts/ingest.py first."
    )

# LLM selection
if os.getenv("GOOGLE_API_KEY"):
    llm = init_chat_model(
        "gemini-2.5-pro", model_provider="google_genai"
    )
elif os.getenv("OPENAI_API_KEY"):
    llm = init_chat_model(
        "gpt-3.5-turbo", model_provider="openai"
    )
else:
    raise RuntimeError("Set either OPENAI_API_KEY or GOOGLE_API_KEY")

# RAG prompt from hub
prompt = hub.pull("rlm/rag-prompt")

# structured schema for the analyzed query
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query."
    ]

# full pipeline state
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# analyze the raw question into a Search struct
def analyze_query(state: State):
    # this helper enforces your Search schema
    structured = llm.with_structured_output(Search)
    return {"query": structured.invoke(state["question"])}

# retrieve, filtering by section
def retrieve(state: State):
    q = state["query"]
    docs = vector_store.similarity_search(q["query"], k=8)
    # manual metadata filter (FAISS has no native filter)
    docs = [
        d for d in docs if d.metadata.get("section") == q["section"]
    ][:4]
    return {"context": docs}

# generate final answer 
def generate(state: State):
    ctxt = "\n\n".join(d.page_content for d in state["context"])
    msgs = prompt.invoke(
        {"question": state["question"], "context": ctxt}
    )
    return {"answer": llm.invoke(msgs).content}


builder = StateGraph(State).add_sequence(
    [analyze_query, retrieve, generate]
)
builder.add_edge(START, "analyze_query")
graph = builder.compile()