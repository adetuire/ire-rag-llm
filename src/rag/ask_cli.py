# src/rag/ask_cli.py
import os, time, json, typer
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from .vector import vector_store  

app = typer.Typer(add_completion=False)

def _approx_tokens(text: str) -> int:
    # Cheap token estimate without adding deps
    # ~4 chars/token heuristic
    return max(1, round(len(text) / 4))

@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="User question"),
    k: int = typer.Option(4, help="Top-k documents"),
    temperature: float = typer.Option(0.2),
):
    # Retrieve
    vs = vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    t0 = time.perf_counter()
    docs = retriever.get_relevant_documents(query)
    t1 = time.perf_counter()

    context = "\n\n".join(d.page_content for d in docs)
    ctx_tokens = _approx_tokens(context)

    # Generate
    model = os.getenv("RAG_MODEL", "mistral:7b")
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use the context to answer. If missing, say you don't know.\n\n{context}"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    g0 = time.perf_counter()
    answer = chain.invoke({"context": context, "question": query}).content
    g1 = time.perf_counter()

    # Output + metrics
    print("\n" + answer.strip() + "\n")
    metrics = {
        "k": k,
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": round((g1 - g0) * 1000, 1),
        "context_tokens_approx": int(ctx_tokens),
        "model": model,
        "index_dir": os.getenv("RAG_INDEX_DIR", "<unset>"),
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    app()
