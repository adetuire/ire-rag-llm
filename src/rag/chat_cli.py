# pyright: reportUnusedImport=false, reportMissingTypeStubs=false
# The directives above silence warnings about runtime-only imports and local modules without stubs.

# src/rag/chat_cli.py
import os
import typer
from typing import Any, Dict, List, cast

# Import the conversational chain from the local module.
try:  # pragma: no cover
    import pyreadline3 as _pyreadline3  
    import readline as _readline        
except Exception:
    pass

# Import the conversational chain from the local module.
from .conversational_chain import chain  # type: ignore

app = typer.Typer(add_completion=False)

@app.command()
def main() -> None:
    """
    Simple REPL-style chat using the conversational chain.
    Ctrl+C / Ctrl+D or 'exit' to quit.
    """
    print("Interactive chat. Type 'exit' to quit.\n")

    history: List[Dict[str, str]] = []

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return

        if user.lower() in {"exit", "quit"}:
            print("bye!")
            return
        if not user:
            continue

        # Prepare the state for the chain
        state: Dict[str, Any] = {"messages": history + [{"role": "user", "content": user}]}

        # Pylance type checking: ensure chain is callable
        resp = cast(Any, chain).invoke(cast(Any, state))
        content = getattr(resp, "content", resp)

        if not isinstance(content, str):
            content = str(content)

        print(f"bot> {content}\n")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    app()
