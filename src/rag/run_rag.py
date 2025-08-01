#!/usr/bin/env python3
import argparse
from rag.chain_rag import graph    # from your package entry

def main():
    p = argparse.ArgumentParser(prog="run-rag")
    p.add_argument("-q","--question", required=True)
    p.add_argument("-k","--k", type=int, default=4)
    args = p.parse_args()

    print(f"[run-rag] question={args.question!r}")
    result = graph.invoke({"question": args.question})
    print("\n---- QUERY ----\n", result["query"])
    print("\n---- CONTEXT ----")
    for d in result["context"]:
        print(f"> ({d.metadata['section']}) {d.page_content[:200]}…")
    print("\n---- ANSWER ----\n", result["answer"])

if __name__=="__main__":
    main()
