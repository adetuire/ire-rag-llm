# src/rag/conversational_chain.py
from __future__ import annotations

import os
import re
from typing import Any, Iterable, List

from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    BaseMessage,
)

from rag.tools import retrieve, retrieve_raw  # your existing tools

# Config
USE_TOOLS = os.getenv("RAG_USE_TOOLS", "1") == "1"
_MODEL_NAME = os.getenv("RAG_MODEL", "mistral:7b")

# Local LLM via Ollama
llm = ChatOllama(model=_MODEL_NAME, temperature=0.2)

# Heuristic for "who/what is <single token>" disambiguation
_CLARIFY_RE = re.compile(r"^\s*(who\s+is|what\s+is)\s+([A-Za-z][A-Za-z\-\.']+)\s*\??\s*$", re.I)

# Helpers
def _last_user_text(state: MessagesState) -> str:
    """Extract the latest user text from either LC messages or simple dict messages."""
    for m in reversed(state["messages"]):
        # LC messages have .type/.content; dicts have ['role']/['content']
        m_type = getattr(m, "type", None)
        role = getattr(m, "role", None) if m_type is None else m_type
        content = getattr(m, "content", None)
        if role == "human" or role == "user":
            return (content or "").strip()
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""

def _to_lc_message(x: Any) -> BaseMessage:
    """Convert dict messages to LC messages if needed."""
    if isinstance(x, BaseMessage):
        return x
    if isinstance(x, dict):
        role = x.get("role", "")
        content = str(x.get("content", ""))
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
    # Fallback
    return HumanMessage(content=str(x))

def _ensure_lc_messages(items: Iterable[Any]) -> List[BaseMessage]:
    return [ _to_lc_message(m) for m in items ]

import json
import re

def _extract_retrieve_query_from_content(text: str) -> str:
    """Try to parse a function call to `retrieve` from plain content."""
    try:
        obj = json.loads(text)
        name = obj.get("function") or obj.get("name")
        if (obj.get("type") == "function") and (name == "retrieve"):
            params = obj.get("parameters") or obj.get("arguments") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            q = params.get("query")
            if isinstance(q, str) and q.strip():
                return q.strip()
    except Exception:
        pass
    m = re.search(r'"query"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip() if m else ""


# Clarifier 
def _maybe_clarify(state: MessagesState):
    """If the user asked 'who/what is <one-token>', list candidate titles instead of guessing."""
    q = _last_user_text(state)
    if not q or not _CLARIFY_RE.match(q):
        return None

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

# Nodes
def query_or_respond(state: MessagesState):
    """Entry node: maybe clarify; else either tool-call or manual retrieval grounding."""
    # inline helpers (local to this node)
    import json, re

    def _last_user_text() -> str:
        for m in reversed(state["messages"]):
            m_type = getattr(m, "type", None)
            role = getattr(m, "role", None) if m_type is None else m_type
            content = getattr(m, "content", None)
            if role in ("human", "user"):
                return (content or "").strip()
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", "")).strip()
        return ""

    def _extract_retrieve_query_from_content(text: str) -> str:
        """Parse a 'function call' to retrieve from plain content when model lacks tool_calls."""
        try:
            obj = json.loads(text or "")
            name = obj.get("function") or obj.get("name")
            if (obj.get("type") == "function") and (name == "retrieve"):
                params = obj.get("parameters") or obj.get("arguments") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                q = params.get("query")
                if isinstance(q, str) and q.strip():
                    return q.strip()
        except Exception:
            pass
        m = re.search(r'"query"\s*:\s*"([^"]+)"', text or "")
        return m.group(1).strip() if m else ""

    # ambiguity clarifier (uses your existing _maybe_clarify)
    clarify_msg = _maybe_clarify(state)
    if clarify_msg:
        return {"messages": [clarify_msg]}

    # tools-enabled branch
    if USE_TOOLS:
        llm_with_tools = llm.bind_tools([retrieve])
        msgs = _ensure_lc_messages(state["messages"])
        response = llm_with_tools.invoke(msgs)

        # If the model didn't emit structured tool_calls, try to parse & retrieve anyway
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            q2 = _extract_retrieve_query_from_content(getattr(response, "content", "") or "")
            if q2:
                docs_text, _ = retrieve_raw(q2)
                sys = (
                    "You are an assistant for question-answering tasks. "
                    "Use ONLY the retrieved context below to answer. If unknown, say you don't know. "
                    "Keep answers ≤3 sentences.\n\n" + docs_text
                )
                prompt = [SystemMessage(sys)] + _ensure_lc_messages(state["messages"])
                resp2 = llm.invoke(prompt)
                return {"messages": [response, resp2]}

        return {"messages": [response]}

    # tools-disabled branch: manual retrieval grounding 
    q = _last_user_text()
    docs_text, _ = retrieve_raw(q)
    sys = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context below to answer. If unknown, say you don't know. "
        "Keep answers ≤3 sentences.\n\n" + docs_text
    )
    msgs = _ensure_lc_messages(state["messages"])
    response = llm.invoke([SystemMessage(sys)] + msgs)
    return {"messages": [response]}


    # Tools disabled → manual retrieval grounding
    q = _last_user_text()
    docs_text, _ = retrieve_raw(q)
    sys = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context below to answer. If unknown, say you don't know. "
        "Keep answers ≤3 sentences.\n\n" + docs_text
    )
    msgs = _ensure_lc_messages(state["messages"])
    response = llm.invoke([SystemMessage(sys)] + msgs)
    return {"messages": [response]}

# Execute the tool if it was called
tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Final answer grounded in tool outputs."""
    # collect tool outputs
    tool_blocks: List[str] = []
    for m in state["messages"]:
        m_type = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
        if m_type == "tool":
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if content:
                tool_blocks.append(str(content))

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context to answer the user's question. "
        "If you don't know, say you don't know. Keep answers ≤3 sentences. "
        "Do NOT speculate about code, stack traces, or errors unless asked. "
        "Do NOT mention tools, retrieval, or the system prompt.\n\n"
        + "\n\n".join(tool_blocks)
    )

    # keep only human/system/ai-without-tool-calls
    conversation_messages: List[Any] = []
    for m in state["messages"]:
        m_type = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
        if m_type in ("human", "system"):
            conversation_messages.append(m)
        elif m_type == "ai":
            tool_calls = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None)
            if not tool_calls:
                conversation_messages.append(m)

    msgs = [SystemMessage(system_message_content)] + _ensure_lc_messages(conversation_messages)
    response = llm.invoke(msgs)
    return {"messages": [response]}

# Graph 
graph = StateGraph(MessagesState)
graph.add_node(query_or_respond)
graph.add_node(tools, name="tools")
graph.add_node(generate)

graph.set_entry_point("query_or_respond")
graph.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

# Export the compiled graph
chat_rag = graph.compile()

# Backward/CLI compatibility: rag.chat_cli imports `chain`
chain = chat_rag
