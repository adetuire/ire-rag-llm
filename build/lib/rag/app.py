from fastapi import FastAPI, Query
from src.rag.chain2 import retrieve_v2

app = FastAPI(title="RAG Retrieval API")

@app.get("/retrieve")
def retrieve(q: str = Query(..., description="Your question"), k: int = 4):
    return {"chunks": retrieve_v2(q, k)}