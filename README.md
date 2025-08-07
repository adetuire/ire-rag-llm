# IRE-RAG-LLM
##  Version 1.2.0 — Conversational RAG (LangGraph + local Ollama + FAISS)

This repo contains a **conversational** RAG pipeline (v1.2) and **legacy single-turn** RAG (v1).  
v2 runs fully local by default: **Ollama** for the LLM and **Hugging Face sentence-transformers** for embeddings.

---

## What’s in v2?

| Area | v2 (1.2.0) |
|---|---|
| LLM | Local **Ollama** (`mistral:7b` by default) |
| Retrieval store | FAISS (Hugging Face `all-MiniLM-L6-v2`) |
| Chat orchestration | **LangGraph** with optional tool-calling |
| API | **FastAPI** `POST /chat` |
| CLI | `rag-chat` terminal chat (zero network) |
| Extras | **Ambiguity clarification** (e.g., “who is ada?” -> show candidates) |
| Config flags | `RAG_USE_TOOLS`, `RAG_INDEX_DIR` |

**Ambiguity clarification**: if a user asks a short, ambiguous query (e.g., `who is ada?`), the graph first **retrieves top candidates** and asks you to choose instead of guessing.  
Set `RAG_USE_TOOLS=0` to disable tool-based retrieval and fall back to plain prompting.

---

## Repo layout (v1.2.0)

# scripts/
    wiki_extract.py # optional: extract wiki dump -> txt files
    
    build_index.py # build FAISS from a folder of txt files

# src/rag/
    app.py # FastAPI app (POST /chat)
    
    chat_cli.py # terminal chat CLI (rag-chat)
    
    conversational_chain.py# LangGraph: query→tools→generate
    
    tools.py # @tool retrieve(...) for LangGraph
    
    vector.py # FAISS + HF embeddings (cached)

# data/
    faiss_blog/ # sample index (blog)
    
    faiss_wiki/ # optional wiki index (if you build it)
    
    wiki_docs/ # optional extracted wiki pages (txt)

---

## Install

```bash
git clone https://github.com/adetuire/ire-rag-llm.git
cd ire-rag-llm
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e .
```

## Build a FAISS index (choose ONE)
# A. Demo blog (quickest)

    Builds a small FAISS index from Lilian Weng’s blog.

```bash
python scripts/build_index.py  # writes to data/faiss_blog/
export RAG_INDEX_DIR=data/faiss_blog        # (Git Bash / Linux / macOS)
# PowerShell: $env:RAG_INDEX_DIR="data/faiss_blog"
```

# B. Wikipedia (offline, local)

    1. Download Simple English wiki dump (≈ 300 MB):

```bash
mkdir -p dumps
curl -L -o dumps/simplewiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
```

    2. Extract a subset to txt files:

```bash
python scripts/wiki_extract.py \
  --dump dumps/simplewiki-latest-pages-articles.xml.bz2 \
  --out data/wiki_docs \
  --limit 20000
```

    3. Index the extracted packages

```bash
python scripts/build_index.py --docs data/wiki_docs --out data/faiss_wiki
export RAG_INDEX_DIR=data/faiss_wiki
# PowerShell: $env:RAG_INDEX_DIR="data/faiss_wiki"
```

    Embeddings: free, CPU-only (sentence-transformers/all-MiniLM-L6-v2).
    
    Files are stored under data/….

## Run locally
# Option 1 - API server

    1. Start Ollama and ensure a tools-capable model is available

```bash
ollama serve   # keep it running
ollama pull mistral:7b      # default in code; needs ~4–5 GiB free RAM
```

    2. Run the API

```bash
uvicorn src.rag.app:app --reload --port 8000
```

    3. Call the endpoint

        a. Git Bash / macOS / Linux (curl):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"history": [], "message": "Explain reflection in autonomous agents."}'
```

        b. Windows PowerShell:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"history": [], "message": "Explain reflection in autonomous agents."}'
```

# Option 2 - Terminal chat (CLI)
```bash
rag-chat
# or: python src/rag/chat_cli.py
```

    Example:

```bash
Conversational RAG (Ctrl-C to quit)
you ▸ who is ada?
rag ▸ Your question is ambiguous. Did you mean one of these?
- Lawrence Higby
- Adana massacre
- ...
you ▸ Ada Lovelace
rag ▸ Ada Lovelace (1815–1852) …
```

## Configuration
| Variable | What it does	 | Git Bash / macOS / Linux	 | PowerShell |
|---------|------------------|-----------------------|--------------------|
| **RAG_INDEX_DIR**	 | Path to an existing FAISS index directory | **export RAG_INDEX_DIR=data/faiss_wiki**  | **$env:RAG_INDEX_DIR="data/faiss_wiki"** |
| **RAG_USE_TOOLS** | **1** = let the model call the **retrieve** tool; **0** = disable tool calls | **export RAG_USE_TOOLS=1** | **$env:RAG_USE_TOOLS="1"** |
| **OLLAMA_HOST** | custom Ollama host | **export OLLAMA_HOST=http://localhost:1143** | **$env:OLLAMA_HOST="http://localhost:11434"** |


## Troubleshooting
| Error message                                                         | Likely cause                                      | Fix (Git Bash / macOS / Linux)                                                                              | Fix (PowerShell)                                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `… does not support tools (status 400)`                               | Model doesn’t support tool calls                  | **`export RAG_USE_TOOLS=0`** (or use a tools-capable instruct model)                                        | **`$env:RAG_USE_TOOLS="0"`**                                                       |
| `model requires more system memory (X GiB) than is available (Y GiB)` | Model too large for RAM                           | Pull a smaller/quantized model (e.g., **`ollama pull mistral:7b-instruct-q4_K_M`**), close apps, or add RAM | Same as Bash                                                                       |
| `functools._lru_cache_wrapper has no attribute 'as_retriever'`        | Calling the cached function instead of its return | Ensure you call **`vector_store()`** (not `vector_store`); update to latest code                            | Same as Bash                                                                       |
| `FutureWarning: encoder_attention_mask`                               | HF transformers deprecation notice                | Safe to ignore (or upgrade transformers)                                                                    | Safe to ignore                                                                     |
| PowerShell `curl` → `Internal Server Error`                           | PS `curl` is an alias to `Invoke-WebRequest`      | Use Git Bash `curl`                                                                                         | Use **`Invoke-WebRequest`** or **`curl.exe`** explicitly                           |
| `Error: listen tcp 127.0.0.1:11234: bind: … in use`                   | Another Ollama is running on 11234(example)                | **`pkill ollama`** (or restart the Ollama service)                                                          | **`Stop-Process -Name ollama -Force`** (or change port via **`$env:OLLAMA_HOST`**) |

## old version below
| Feature | v1.0.0 (single-turn) | v1.1.1 |
|---------|------------------|-----------------------|
| Retrieval store | FAISS index of Lilian Weng blog | same index  |
| LLM | any OpenAI / Gemini | **Local Ollama (default `mistral:7b`)** |
| Memory | none | **Conversational context** via LangGraph |
| API | GET `/retrieve` | **POST `/chat`** (multi-turn) |

### Quick start

```bash
git clone ...
cd ire-rag-llm
python -m venv .venv && source .venv/Scripts/activate  # Win: Activate

pip install -e .            # installs v2 deps
python scripts/build_index.py   # one-time: downloads & indexes the blog post

ollama serve &               # make sure Mistral-7B is pulled
uvicorn src.rag.app:app --reload --port 8000
```

## Then:

    curl -X POST http://localhost:8000/chat \
         
         -H "Content-Type: application/json" \
         
         -d '{"history": [], "message": "What is task decomposition?"}'

## RAM note
    mistral:7b needs >=4.2 GiB free.
    
    If that’s tight, pull llama3:8b-instruct-q4_K_M (you need a model that has tools-enabled) and
    
    change one line in src/rag/conversational_chain.py:

        "llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.2)"

## Offline Run(hf + ollama)
```bash
    "No cloud keys, no OpenAI fees, everything happens on your laptop."
```
## 1.  Install the extra wheels
```bash
    pip install -U langchain-ollama langchain-huggingface ollama
```
## 2. use a tools-enabled model based on your memory size
```bash
    ollama pull mistral:7b-instruct
```
## 3. Build the FAISS index once
```bash
    python scripts/build_index.py

```
    This downloads Lilian Weng’s RAG blog post, chunks it, embeds with the free
    
    sentence-transformers/all-MiniLM-L6-v2 model and writes
    
    data/faiss_blog/.

## 4. Start the lacal ollama
```bash
    ollama serve &   # keep running in a seperate tab
    uvicorn src.rag.app:app - reload --port 8000
```
## 5. Now you can use it
```bash
    # one-off curl
    curl -X POST http://localhost:8000/chat \
    
         -H "Content-Type: application/json" \
    
         -d '{"history": [], "message": "Hi 👋"}'

    # two-turn example
    curl -X POST http://localhost:8000/chat \
        
         -H "Content-Type: application/json" \
        
         -d '{"history": [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello! How can I help?"}], "message": "What is task decomposition?"}'
```
    Ollama streams the tokens in the terminal running uvicorn;  
    
    curl returns a compact JSON answer:
```bash
    {
        "answer": "Task decomposition means breaking a complex objective into smaller actions so an autonomous agent can solve each one in turn..."
    }
```

---

## Command-line chat

If you prefer a zero-network terminal chat, use the built-in CLI.

### Run

```bash
    python src/rag/chat_cli.py

    After running: "pip install -e ."

    run: rag-chat
```

## Example of a session
```bash
        Conversational RAG (Ctrl-C to quit)
    you ▸ Hi
    rag ▸ Hello! How can I help?

    you ▸ What is task decomposition?
    rag ▸ Task decomposition is the practice of breaking a complex goal …
```

## V1.0.0 README below
## Offline Run(hf + ollama)
```bash
    "No cloud keys, no OpenAI fees, everything happens on your laptop."
```
## 1.  Install the extra wheels
```bash
    pip install -U langchain-ollama langchain-huggingface ollama
```
## 2. use a tools-enabled model based on your memory size
```bash
    ollama pull mistral:7b-instruct
```
## 3. Build the FAISS index once
```bash
    python scripts/build_index.py

```
    This downloads Lilian Weng’s RAG blog post, chunks it, embeds with the free
    
    sentence-transformers/all-MiniLM-L6-v2 model and writes
    
    data/faiss_blog/.

## 4. Start the lacal ollama
```bash
    ollama serve &   # keep running in a seperate tab
    uvicorn src.rag.app:app - reload --port 8000
```
## 5. Now you can use it
```bash
    # one-off curl
    curl -X POST http://localhost:8000/chat \
    
         -H "Content-Type: application/json" \
    
         -d '{"history": [], "message": "Hi 👋"}'

    # two-turn example
    curl -X POST http://localhost:8000/chat \
        
         -H "Content-Type: application/json" \
        
         -d '{"history": [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello! How can I help?"}], "message": "What is task decomposition?"}'
```
    Ollama streams the tokens in the terminal running uvicorn;  
    
    curl returns a compact JSON answer:
```bash
    {
        "answer": "Task decomposition means breaking a complex objective into smaller actions so an autonomous agent can solve each one in turn..."
    }
```

---

## Command-line chat

If you prefer a zero-network terminal chat, use the built-in CLI.

### Run

```bash
    python src/rag/chat_cli.py

    After running: "pip install -e ."

    run: rag-chat
```

## Example of a session
```bash
        Conversational RAG (Ctrl-C to quit)
    you ▸ Hi
    rag ▸ Hello! How can I help?

    you ▸ What is task decomposition?
    rag ▸ Task decomposition is the practice of breaking a complex goal …
```

## V1.0.0 README below
    Legacy single-turn RAG scripts now live in src/rag/legacy/; the main package contains only the v2 conversational pipeline.

### V1.0.0
### V1.0.0

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
"source .venv/Scripts/activate"
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

        Asuume you have dropped a .html/.md/.txt file in data/raw/ 
        
        or you have ran another version of ingest that populates that folder with the needed files. 
        
        If the folder is empty it raises the "No documents found"

4.  Run the ingester with "python scripts/ingest3.py"
    
    This will:

        pull Lilian Weng’s “LLM-Powered Autonomous Agents” blog post directly from the web (no files needed in data/raw/);

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
### Script Guide & Quick-start Cheatsheet

| Script / CLI entry-point | What it does | required before running | Example |
| ------------------------ | ------------ | ------------------- | ------- |
| `python scripts/ingest.py` | Index **any local `.html / .md / .txt` files** in `data/raw/` into `data/faiss_index.faiss` + `data/store.pkl`. | Put files in `data/raw/` first. No API keys. | `python scripts/ingest.py` |
| `python scripts/ingest2.py` | **One-shot downloader**: fetches Lilian Weng’s “LLM Agents” blog post, chunks it and builds `data/faiss_index_v2.faiss`. | Internet connection. No API keys. | `python scripts/ingest2.py` |
| `python scripts/ingest3.py` | Same as above but uses a manual FAISS build + pickles the doc-store (`data/faiss_index_v3.faiss`, `data/store.pkl`). | Internet. No API keys. | `python scripts/ingest3.py` |
| `python -m src.rag.chain -q "..." [--k N]` | **Pure retrieval v1** – returns the *k* nearest chunks from `faiss_index.faiss`. | Run **`ingest.py`** first. | `python -m src.rag.chain -q "Explain X" --k 3` |
| `python -m src.rag.chain2 -q "..." [--k N]` | **Pure retrieval v2** – uses `faiss_index_v2.faiss`. | Run **`ingest2.py`** first. | `python -m src.rag.chain2 -q "Explain X"` |
| `run-retrieval -q "..." [-k N]` | CLI wrapper for **`chain2.retrieve_v2()`** (identical output). Installed automatically via `pip install -e .`. | Same as above. | `run-retrieval -q "Explain X" -k 5` |
| `run-rag -q "..."` | Full **RAG pipeline** – query-analysis to filtered retrieval to LLM answer (LangGraph). Uses `chain_rag.py`. | 1. **`OPENAI_API_KEY` _or_ `GOOGLE_API_KEY`** in env  <br>2. `ingest.py` or `ingest3.py` done. | `export OPENAI_API_KEY=…`<br>`run-rag -q "What is Task Decomposition?"` |
| `uvicorn rag.app:app --reload --port 8000` | Spins up a **FastAPI** micro-service.<br>GET `/retrieve?q=<question>&k=<int>` returns JSON with top-k chunks (powered by `chain2`). | `ingest2.py` done. | `curl "http://localhost:8000/retrieve?q=Explain+X&k=3"` |

### Environment variables

* `OPENAI_API_KEY` – set for OpenAI Chat completion in `chain_rag.py`.
* `GOOGLE_API_KEY` – alternative: Gemini 2.5 Pro via `langchain-google-genai`.
* `USER_AGENT` – polite header for web scrapers (scripts set a default).
---