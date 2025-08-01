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

# load the store we built above
with open("data/store.pkl", "rb") as f:
    vector_store: InMemoryVectorStore = pickle.load(f)

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
