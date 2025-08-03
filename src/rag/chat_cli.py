#!/usr/bin/env python
"""
Conversational RAG – terminal chat.

$ python scripts/chat_cli.py
"""
import json, sys, typer
import pyreadline3 as readline
from rag.conversational_chain import chat_rag

app = typer.Typer(add_completion=False)
history: list[dict] = []          # running dialogue


@app.command()
def chat():
    typer.echo("Conversational RAG (Ctrl-C to quit)")
    try:
        while True:
            user = input("you ▸ ").strip()
            if not user:
                continue
            history.append({"role": "user", "content": user})

            state = {"messages": history}
            result = chat_rag.invoke(state)
            assistant = result["messages"][-1].content
            print("rag ▸", assistant)

            history.append({"role": "assistant", "content": assistant})
    except (EOFError, KeyboardInterrupt):
        typer.echo("\nbye!!!")


if __name__ == "__main__":
    app()
