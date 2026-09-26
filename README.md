# Ask My Docs

A domain-specific question-answering system over a set of documents, with
hybrid retrieval (BM25 + vector search), cross-encoder reranking,
citation enforcement, and a CI-gated evaluation pipeline.

## Project layout

- `data/raw/`   - the source documents
- `src/`        - the application code (ingestion, retrieval, answering)
- `eval/`       - test questions and evaluation scripts
- `scripts/`    - helper scripts, like the setup check

## Setup

1. Create and activate a virtual environment:
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - Windows:   `python -m venv .venv` then `.venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`
4. Check everything works: `python scripts/check_setup.py`
