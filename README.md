# IRE-RAG-LLM

A local Retrieval-Augmented Generation (RAG) demo using FAISS + HuggingFace embeddings (free) and an optional LLM-API pipeline.  
Supports:
- `scripts/ingest.py` -> load & chunk HTML/TXT in `data/raw/` -> build FAISS index  
- `run-retrieval` CLI -> pure-retrieval over FAISS  
- `run-rag` CLI -> full RAG QA using prompt hub  
- FastAPI -> REST endpoint at `GET /qa?question=...&k=...`

---

## Install

```bash
"git clone https://github.com/adetuire/ire-rag-llm.git"
cd ire-rag-llm

# create virtualenv
"python -m venv .venv"
# Windows: ".venv\Scripts\activate"
# macOS/Linux: "source .venv/bin/activate"

"pip install -r requirements.txt"
# or for editable install:
# "pip install -e ."


## Prepare & Index
1.  Drop any .html, .md, or .txt docs into data/raw/.

2.  Run the ingester:
        "python scripts/ingest.py"
    This will:

        Parse each file with BeautifulSoup

        Chunk with RecursiveCharacterTextSplitter

        Embed with HuggingFaceEmbeddings

        Build & save data/faiss_index.faiss


## Pure Retrieval CLI
Query the FAISS index without hitting any paid API:
    "run-retrieval --question "Explain task decomposition" --k 4"


## RAG QA CLI
If you’ve built the FAISS index and have an API key in your environment (OpenAI or Google), you can run full RAG:
    "run-rag --question "What is Task Decomposition?" --k 3"

## FastAPI Server
Start a local server (defaults to port 8000):
    "uvicorn src.rag.app:app --reload --port 8000"

Endpoints
    GET /retrieve?question=...&k=... -> returns top-k chunks
    POST /rag with JSON {"question": "...", "k": 3} -> returns RAG answer

## Configuration
.env file in project root (ignored by git) can hold:
    "OPENAI_API_KEY=..."
    "GOOGLE_API_KEY=..."
.gitignore already excludes .env, .venv/, cache, data outputs, etc.