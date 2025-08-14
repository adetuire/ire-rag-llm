# src/rag/chat_cli.py
# pyright: reportMissingTypeStubs=false
import typer
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from .conversational_chain import chain  # the compiled graph

app = typer.Typer(add_completion=False)

@app.command()
def main() -> None:
    print("Interactive chat. Type 'exit' to quit.\n")
    history: List[BaseMessage] = []

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

        history.append(HumanMessage(content=user))

        # LangGraph returns {"messages": [...]}
        out = chain.invoke({"messages": history})
        msgs = out.get("messages", []) if isinstance(out, dict) else []
        last = None
        if isinstance(msgs, list) and msgs:
            # prefer last AI message if present
            for m in reversed(msgs):
                if getattr(m, "type", None) == "ai":
                    last = m
                    break
            if last is None:
                last = msgs[-1]
        else:
            last = out  # fallback

        content = getattr(last, "content", str(last))
        if not isinstance(content, str):
            content = str(content)

        print(f"bot> {content}\n")
        history.append(AIMessage(content=content))

if __name__ == "__main__":
    app()
