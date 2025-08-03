from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.conversational_chain import chat_rag

app = FastAPI(title="Conversational RAG Demo")

class ChatRequest(BaseModel):
    history: list[dict]   # [{'role':'user','content':'...'}, ...]
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    state = {"messages": req.history + [{"role":"user","content":req.message}]}
    result = chat_rag.invoke(state)
    return {"answer": result["messages"][-1].content}
