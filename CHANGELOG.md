# Changelog

All notable changes to **ire-rag-llm** will be documented in this file.

This project adheres to **[Semantic Versioning](https://semver.org/)** and the
format is inspired by **[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)**.

---

## [1.3.1] — 2025-08-14

### Fixed
- **`rag-chat` output**: prints only the assistant’s final answer (no more raw graph dicts).

### Added
- **Tiny-model tools fallback**: when small Ollama models (e.g., `llama3.2:1b-instruct-q4_K_M`) _don’t_ emit structured `tool_calls`, we now parse the JSON-like “function” text and still run retrieval → grounded answer.
- **Troubleshooting tips**:
  - Suppress noisy `FutureWarning`s from encoder libs via `PYTHONWARNINGS=ignore::FutureWarning`.

### Changed
- `conversational_chain.py`:
  - Hardened the **tools-enabled path** with inline helpers to parse faux function calls.
  - Preserved LangGraph flow and `chain` export for CLI compatibility.

### Docs
- README clarifications:
  - When to set `RAG_USE_TOOLS=0` (most reliable on tiny models) vs `RAG_USE_TOOLS=1` (structured tool-calls).
  - Quick-start snippets kept Windows/Bash parity.

### Upgrade Notes
```bash
git pull
# if version is pinned locally:
pip install -e .
# or from a fresh environment:
python -m venv .venv && source .venv/bin/activate  # (PowerShell: .\.venv\Scripts\Activate.ps1)
pip install -e .
```
## [1.3.0] — 2025-08-14
### Added

    Reproducible 10k Simple Wikipedia index:

        scripts/wiki_extract.py and scripts/build_index.py

        Windows + macOS/Linux steps; RAG_INDEX_DIR env var

    CLI one-shot QA: rag-ask --query "..." --k 4 prints answer + metrics JSON.

    Interactive chat: rag-chat REPL over the same FAISS index.

    Packaging: pip install -e . exposes rag-ask and rag-chat via console_scripts.

    CPU-only embeddings by default (all-MiniLM-L6-v2); tiny local LLM via Ollama is optional.

    Retrieval-only fallback when no local LLM is available (top doc snippet + metrics).

### Docs

    Expanded README:

        “10k wiki” path and a minimal blog demo (data/faiss_blog) for low-resource testing.

        Environment examples (RAG_INDEX_DIR, RAG_MODEL) for Bash and PowerShell.

        Windows steps (no make dependency).

### Notes

    No breaking changes; existing RAG_INDEX_DIR usage remains the same.


## [1.2.0] - 2025-08-06
### Added
- `rag-chat` CLI (local conversational RAG).
- Env flags: `RAG_USE_TOOLS`, `RAG_INDEX_DIR`.
- Wikipedia offline extract + FAISS index scripts.
- Ambiguity clarification (“who is ada?” → show candidates).

### Fixed
- Tool invocation and retriever wiring issues.
