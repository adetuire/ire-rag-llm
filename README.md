# RAG + vLLM demo

GETTING STARTED
pip install -r requirements.txt

python scripts/ingest2.py

python -m src.rag.run_retrieval -q "..." -k 3
