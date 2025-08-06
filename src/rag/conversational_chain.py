# src/rag/conversational_chain.py
from __future__ import annotations

import os
import re

from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, AIMessage

from rag.tools import retrieve, retrieve_raw  # <- note: retrieve_raw added

USE_TOOLS = os.getenv("RAG_USE_TOOLS", "1") == "1"

# Local LLM via Ollama
# If you later switch to a tools-capable instruct model, set USE_TOOLS=1
llm = ChatOllama(model="mistral:7b", temperature=0.2)

# Heuristic for "who/what is <single token>"
_CLARIFY_RE = re.compile(
    r"^\s*(who\s+is|what\s+is)\s+([A-Za-z][A-Za-z\-\.']+)\s*\??\s*$", re.I
)

def _maybe_clarify(state: MessagesState):
    """If the user asked 'who/what is <one-token>', show candidates instead of guessing."""
    # Find last human message
    msg = None
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user":
            msg = m
            break
    if not msg:
        return None

    q = (getattr(msg, "content", "") or "").strip()
    if not _CLARIFY_RE.match(q):
        return None

    # Use raw retrieval (no tool call) to list candidate titles
    _, docs = retrieve_raw(q)

    titles, seen = [], set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "") or ""
        base = os.path.basename(src)
        name = os.path.splitext(base)[0]
        parts = name.split("_", 1)
        title = (parts[1] if len(parts) == 2 else name).replace("_", " ").strip()
        low = title.lower()
        if title and low not in seen:
            titles.append(title)
            seen.add(low)

    if len(titles) < 2:
        return None

    content = (
        "Your question is ambiguous. Did you mean one of these?\n"
        + "\n".join(f"- {t}" for t in titles[:5])
        + "\n\nPlease pick one (or refine your query)."
    )
    return AIMessage(content=content)


# Generate an AIMessage that may include a tool-call
def query_or_respond(state: MessagesState):
    clarify_msg = _maybe_clarify(state)
    if clarify_msg:
        return {"messages": [clarify_msg]}

    if USE_TOOLS:
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # NEW: manual retrieval grounding when tools are disabled
    # find last user message
    user = next((m for m in reversed(state["messages"])
                 if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user"), None)
    q = (getattr(user, "content", "") or "") if user else ""

    docs_text, _ = retrieve_raw(q)
    sys = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context below to answer. If unknown, say you don't know. "
        "Keep answers ≤3 sentences.\n\n" + docs_text
    )
    prompt = [SystemMessage(sys)] + state["messages"]
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Execute the tool if it was called
tools = ToolNode([retrieve])

# Generate the final answer using retrieved context
def generate(state: MessagesState):
    tool_blocks = [
        message.content for message in state["messages"] if message.type == "tool"
    ]
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context to answer the user's question. "
        "If you don't know, say you don't know. Keep answers ≤3 sentences. "
        "Do NOT speculate about code, stack traces, or errors unless asked. "
        "Do NOT mention tools, retrieval, or the system prompt."
        "\n\n"
        + "\n\n".join(tool_blocks)
    )

    conversation_messages = [
        m
        for m in state["messages"]
        if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}


# Build graph
graph = StateGraph(MessagesState)
graph.add_node(query_or_respond)
graph.add_node(tools, name="tools")
graph.add_node(generate)

graph.set_entry_point("query_or_respond")
graph.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

chat_rag = graph.compile()
