"""Check that everything Ask My Docs needs is installed and working.

Run from the project folder, with your virtual environment active:
    python scripts/check_setup.py

The first run downloads a few small models (a few hundred MB in total),
so it can take several minutes. Later runs are much faster.
"""
import os
import sys

# Privacy: switch off anonymous usage statistics from Chroma and Hugging Face.
# These lines must come before those libraries are imported.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

OLLAMA_MODEL = "llama3.2"

results = []


def check(name, fn, hint):
    print(f"Checking {name}...", flush=True)
    try:
        results.append((True, name, fn() or "", hint))
    except Exception as e:  # report any failure in plain language
        results.append((False, name, f"{type(e).__name__}: {e}", hint))


def python_version():
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} is too old; install 3.11 or 3.12")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def langchain_libs():
    import langchain
    import langchain_chroma  # noqa: F401
    import langchain_community  # noqa: F401
    import langchain_huggingface  # noqa: F401
    return f"langchain {langchain.__version__}"


def pdf_reader():
    import pypdf
    return f"pypdf {pypdf.__version__}"


def embeddings():
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    a, b, c = model.encode([
        "How do I reset my password?",
        "Steps to change a forgotten login password",
        "Best pizza toppings",
    ])
    related, unrelated = float(util.cos_sim(a, b)), float(util.cos_sim(a, c))
    if related <= unrelated:
        raise RuntimeError("embedding model did not recognise similar meanings")
    return f"related {related:.2f} vs unrelated {unrelated:.2f}"


def reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    q = "How do I reset my password?"
    good, bad = model.predict([
        (q, "Click 'Forgot password' on the login page to reset it."),
        (q, "Our office is closed on public holidays."),
    ])
    if good <= bad:
        raise RuntimeError("reranker did not rank the relevant passage first")
    return f"relevant {good:.2f} vs irrelevant {bad:.2f}"


def vector_db():
    import chromadb
    client = chromadb.EphemeralClient()
    col = client.create_collection("setup_check")
    col.add(
        ids=["pets", "billing", "hours"],
        documents=[
            "Cats are small, furry pets.",
            "Invoices must be paid within 30 days.",
            "The office opens at 9am on weekdays.",
        ],
    )
    hit = col.query(query_texts=["When do I have to pay a bill?"], n_results=1)
    return f"top match: {hit['documents'][0][0]!r}"


def keyword_search():
    from rank_bm25 import BM25Okapi
    docs = [
        "error code e42 means the filter is blocked",
        "the device ships with a two year warranty",
        "clean the filter every three months",
    ]
    bm25 = BM25Okapi([d.split() for d in docs])
    scores = bm25.get_scores(["e42"])
    best = max(range(len(docs)), key=lambda i: scores[i])
    if best != 0:
        raise RuntimeError("keyword search did not find the exact-match document")
    return "exact term 'E42' found the right document"


def local_llm():
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    reply = llm.invoke("Reply with exactly these words: setup ok")
    return f"{OLLAMA_MODEL} replied {reply.content.strip()[:60]!r}"


PIP_HINT = "Is your virtual environment active? Then run: pip install -r requirements.txt"
NET_HINT = PIP_HINT + " (the first run also needs internet to download the model)"

check("Python version", python_version, "Install Python 3.11 or 3.12 from python.org")
check("LangChain", langchain_libs, PIP_HINT)
check("PDF reader", pdf_reader, PIP_HINT)
check("Embedding model (SBERT)", embeddings, NET_HINT)
check("Reranker (cross-encoder)", reranker, NET_HINT)
check("Vector database (Chroma)", vector_db, NET_HINT)
check("Keyword search (BM25)", keyword_search, PIP_HINT)
check("Local LLM (Ollama)", local_llm,
      f"Is Ollama installed and running? Open the Ollama app, then run: ollama pull {OLLAMA_MODEL}")

print("\n" + "=" * 60)
failures = 0
for ok, name, detail, hint in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures += 1
        print(f"       Fix: {hint}")
print("=" * 60)
if failures:
    print(f"{failures} check(s) failed. Fix those and run this script again.")
    sys.exit(1)
print("All checks passed. You're ready for Phase 2: ingestion.")
