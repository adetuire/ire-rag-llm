# src/rag/chain_rag.py
from __future__ import annotations
import os, getpass, pickle
from typing import Literal, List, TypedDict
from typing_extensions import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph

from pathlib import Path
from langchain_community.vectorstores import FAISS

FAISS_FP   = Path("data/faiss_index.faiss")
PICKLE_FP  = Path("data/store.pkl")

if PICKLE_FP.exists():                     # original behaviour
    import pickle, warnings
    with PICKLE_FP.open("rb") as f:
        vector_store = pickle.load(f)
    print("[chain_rag] loaded in-memory pickle store")
elif FAISS_FP.exists():                    # fallback – no pickle, use FAISS file
    from langchain_huggingface import HuggingFaceEmbeddings
    vector_store = FAISS.load_local(
        str(FAISS_FP),
        HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )
    print("[chain_rag] loaded FAISS index")
else:
    raise FileNotFoundError(
        "Neither data/store.pkl nor data/faiss_index.faiss found – run scripts/ingest.py"
    )

# LLM init (same as before)
if os.getenv("GOOGLE_API_KEY"):
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
elif os.getenv("OPENAI_API_KEY"):
    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")
else:
    raise RuntimeError("Set OPENAI_API_KEY or GOOGLE_API_KEY")

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
    structured_llm = llm.with_structured_output(Search)
    q = structured_llm.invoke(state["question"])
    return {"query": q}

# retrieve, filtering by section
def retrieve(state: State):
    q = state["query"]
    docs = vector_store.similarity_search(
        q["query"],
        filter=lambda d: d.metadata.get("section") == q["section"],
        k=4
    )
    return {"context": docs}

# generate final answer 
def generate(state: State):
    ctxt = "\n\n".join(d.page_content for d in state["context"])
    msgs = prompt.invoke({"question": state["question"], "context": ctxt})
    resp = llm.invoke(msgs)
    return {"answer": resp.content}


builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
builder.add_edge(START, "analyze_query")
graph = builder.compile()
