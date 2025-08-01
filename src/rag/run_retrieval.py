#!/usr/bin/env python3
"""
run_retrieval.py

Standalone entry-point for the v2 RAG pipeline.
Invokes retrieve_v2() from chain2.py and prints out the top-k chunks.
"""

import argparse
from src.rag.chain2 import retrieve_v2

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k context chunks for a given question."
    )
    parser.add_argument(
        "--question", "-q", required=True, help="The question to ask the RAG pipeline."
    )
    parser.add_argument(
        "--k", "-k", type=int, default=4, help="Number of chunks to retrieve."
    )
    args = parser.parse_args()

    chunks = retrieve_v2(args.question, k=args.k)
    for i, chunk in enumerate(chunks, 1):
        print(f"--- chunk {i} ---")
        print(chunk)
        print()

if __name__ == "__main__":
    main()
