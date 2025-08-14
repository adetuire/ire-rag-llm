# src/rag/ask_cli.py
import os
import time
import json
import typer
from typing import Any, Dict, List, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from .vector import vector_store  # FAISS loader

app = typer.Typer(add_completion=False)

def _approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))  # ~4 chars/token

def _source_of(doc: Document, i: int) -> str:
    md = doc.metadata or {}
    return md.get("source") or md.get("url") or md.get("path") or f"doc_{i}"

@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="User question"),
    k: int = typer.Option(4, help="Top-k documents"),
    temperature: float = typer.Option(0.2),
    show_sources: bool = typer.Option(False, help="Print the retrieved sources"),
    strict: bool = typer.Option(True, help="Answer only from context; say 'I don't know' if not present"),
) -> None:
    # Retrieve
    vs = vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    t0 = time.perf_counter()
    docs: List[Document] = cast(List[Document], retriever.invoke(query))
    t1 = time.perf_counter()

    context = "\n\n".join(d.page_content for d in docs) if docs else ""
    ctx_tokens = _approx_tokens(context)
    sources: List[str] = [_source_of(d, i) for i, d in enumerate(docs or [])]

    # Try generation (tiny Ollama model); fall back to retrieval-only
    model = os.getenv("RAG_MODEL", "mistral:7b")
    if strict:
        sys_rules = [
            "Use ONLY the provided context to answer.",
            "If the answer is not in the context, reply: I don't know.",
            "Do NOT invent dates, awards, or numbers.",
        ]
    else:
        sys_rules = ["Use the context if helpful; otherwise use your prior knowledge."]

    system_msg = "\n".join(sys_rules) + "\n\n{context}"

    answer: str
    gen_ms = 0.0
    used_model = model

    try:
        llm = ChatOllama(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        g0 = time.perf_counter()
        msg = chain.invoke({"context": context, "question": query})
        content = cast(Any, msg).content  # content can be str | list[...]
        answer = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        g1 = time.perf_counter()
        gen_ms = round((g1 - g0) * 1000, 1)
    except Exception as e:
        used_model = f"retrieval_only ({type(e).__name__})"
        answer = (docs[0].page_content[:500].strip() + " ...") if docs else "No documents found."

    # Output
    print("\n" + answer.strip() + "\n")

    if show_sources and sources:
        print("Sources:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s}")
        print("")

    metrics: Dict[str, Any] = {
        "k": k,
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": gen_ms,
        "context_tokens_approx": int(ctx_tokens),
        "model": used_model,
        "index_dir": os.getenv("RAG_INDEX_DIR", "<Unset>"),
        "strict": strict,
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    app()
