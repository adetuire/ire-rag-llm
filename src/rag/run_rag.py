#!/usr/bin/env python3

import argparse
from rag.chain_rag import graph  # your compiled LangGraph from chain_rag.py

def main():
    p = argparse.ArgumentParser(prog="run-rag",
        description="Run RAG QA over your FAISS index + prompt hub")
    p.add_argument("--question", "-q", required=True, help="The user question")
    p.add_argument("--k",        "-k", type=int, default=4,
                   help="How many chunks to retrieve")
    args = p.parse_args()

    print(f"[run_rag] question={args.question!r}, k={args.k}")
    print("[run_rag] invoking graph…")
    result = graph.invoke({"question": args.question, "k": args.k})
    print("\n---- CONTEXT ----")
    for doc in result["context"]:
        print("> " + doc.page_content.replace("\n", " ")[:200] + "…")
    print("\n---- ANSWER ----")
    print(result["answer"])

if __name__ == "__main__":
    main()