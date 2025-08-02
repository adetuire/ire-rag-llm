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
"python -m venv .venv" # or python3 -m venv .venv
# Windows: 
".venv\Scripts\activate"
# macOS/Linux: 
"source .venv/bin/activate"

# installs the package & console scripts
#for editable install:
"pip install -e ."
# or "pip install -r requirements.txt"



# build the vector store once
# downloads & indexes the blog post
"python scripts/ingest.py"   

# ask something
run-rag --question "What is Task Decomposition?" --k 3

# exit the virtual environment
(.venv) $ deactivate


```

## Prepare & Index
1.  Drop any .html, .md, or .txt docs into data/raw/.

2.  Run the ingester:
        
        "python scripts/ingest.py"

    This will:

        Parse each file with BeautifulSoup

        Chunk with RecursiveCharacterTextSplitter

        Embed with HuggingFaceEmbeddings

        Build & save data/faiss_index.faiss.

3.  Run the ingester with "python scripts/ingester2.py"
    
    This will:

        Asuume you have dropped a .html/.md/.txt file in data/raw/ or you have ran anothe version of ingest that populates that folder with the needed files. If the folder is empty it raises the "No documents found"

4.  Run the ingester with "python scripts/ingest3.py"
    
    This will:

        pull Lilian Weng’s “LLM-Powered Autonomous Agents” blog post directly from the
        web (no files needed in data/raw/);

        chunks it, embeds with Sentence-Transformers (all-MiniLM-L6-v2), and saves:

            data/faiss_index.faiss

            data/store.pkl

        prints a quick progress summary.  


## Pure Retrieval CLI
Query the FAISS index without hitting any paid API:
    
    "run-retrieval --question "Explain task decomposition" --k 4"


## RAG QA CLI
If you’ve built the FAISS index and have an API key in your environment (OpenAI or Google), you can run full RAG:
    
    "run-rag --question "What is Task Decomposition?" --k 3"

## FastAPI Server
Start a local server (defaults to port 8000):
    
    # "uvicorn rag.app:app --reload --port 8000"
    
    # GET /retrieve?question=&k= – top-k chunks

    # POST /rag {"question": "...", "k": 4} – RAG answer

## Endpoints
    GET /retrieve?q=...&k=... -> returns top-k chunks

    e.g after running "uvicorn rag.app:app --reload --port 8000";

    "http://xxx.x.x.x:8000/retrieve?q=Explain%20Task%20Decomposition&k=3"
    
    POST /rag with JSON {"q": "...", "k": 3} -> returns RAG answer

## Configuration
Create a .env alongside pyproject.toml.

.env file in project root (ignored by git) can hold:

    "OPENAI_API_KEY=..."
    "GOOGLE_API_KEY=..."

.gitignore already excludes .env, .venv/, cache, data outputs, etc.



---
Script Guide & Quick-start Cheatsheet

| Script / CLI entry-point | What it does | Needs before running | Example |
| ------------------------ | ------------ | ------------------- | ------- |
| `python scripts/ingest.py` | Index **any local `.html / .md / .txt` files** in `data/raw/` into `data/faiss_index.faiss` + `data/store.pkl`. | Put files in `data/raw/` first. No API keys. | `python scripts/ingest.py` |
| `python scripts/ingest2.py` | **One-shot downloader**: fetches Lilian Weng’s “LLM Agents” blog post, chunks it and builds `data/faiss_index_v2.faiss`. | Internet connection. No API keys. | `python scripts/ingest2.py` |
| `python scripts/ingest3.py` | Same as above but uses a manual FAISS build + pickles the doc-store (`data/faiss_index_v3.faiss`, `data/store.pkl`). | Internet. No API keys. | `python scripts/ingest3.py` |
| `python -m src.rag.chain -q "..." [--k N]` | **Pure retrieval v1** – returns the *k* nearest chunks from `faiss_index.faiss`. | Run **`ingest.py`** first. | `python -m src.rag.chain -q "Explain X" --k 3` |
| `python -m src.rag.chain2 -q "..." [--k N]` | **Pure retrieval v2** – uses `faiss_index_v2.faiss`. | Run **`ingest2.py`** first. | `python -m src.rag.chain2 -q "Explain X"` |
| `run-retrieval -q "..." [-k N]` | CLI wrapper for **`chain2.retrieve_v2()`** (identical output, nicer name). Installed automatically via `pip install -e .`. | Same as above. | `run-retrieval -q "Explain X" -k 5` |
| `run-rag -q "..."` | Full **RAG pipeline** – query-analysis ➜ filtered retrieval ➜ LLM answer (LangGraph). Uses `chain_rag.py`. | 1. **`OPENAI_API_KEY` _or_ `GOOGLE_API_KEY`** in env  <br>2. `ingest.py` or `ingest3.py` done. | `export OPENAI_API_KEY=…`<br>`run-rag -q "What is Task Decomposition?"` |
| `uvicorn rag.app:app --reload --port 8000` | Spins up a **FastAPI** micro-service.<br>GET `/retrieve?q=<question>&k=<int>` returns JSON with top-k chunks (powered by `chain2`). | `ingest2.py` done. | `curl "http://localhost:8000/retrieve?q=Explain+X&k=3"` |

### Environment variables

* `OPENAI_API_KEY` – set for OpenAI Chat completion in `chain_rag.py`.
* `GOOGLE_API_KEY` – alternative: Gemini 2.5 Pro via `langchain-google-genai`.
* `USER_AGENT` – polite header for web scrapers (scripts set a default).
---