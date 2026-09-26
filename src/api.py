"""Web backend: the same answering pipeline, reachable over HTTP.

Run it from the project folder with the virtual environment active:
    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000 in your browser.

Endpoints:
    GET  /          the web page
    GET  /health    is the service up, and how many chunks are indexed
    POST /ask       {"question": "..."} -> answer with checked citations
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from answer import (ANSWER_SCHEMA, MAX_ATTEMPTS, MODEL, PASSAGES_SHOWN,
                    ask_model, parse_reply, validate)
from retrieval import RERANK_MIN_SCORE, Retriever

PROJECT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT / "web"

app = FastAPI(title="Ask My Docs", description="Question answering over the EU AI Act")

# Loaded once when the server starts, not per request: loading the models takes
# a few seconds and they can be reused for every question.
retriever = None
llm = None
# Repeated questions are answered from memory. On a public demo most people ask
# the same few things, so this saves both time and, later, money.
cache = {}
CACHE_LIMIT = 500


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.on_event("startup")
def load_everything():
    global retriever, llm
    from langchain_ollama import ChatOllama

    print("Loading retriever (this takes a few seconds)...")
    retriever = Retriever()
    try:
        llm = ChatOllama(model=MODEL, temperature=0, format=ANSWER_SCHEMA)
    except Exception:
        llm = ChatOllama(model=MODEL, temperature=0, format="json")
    print(f"Ready: {len(retriever.chunks)} chunks indexed.")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(retriever.chunks) if retriever else 0}


@app.get("/")
def home():
    page = WEB_DIR / "index.html"
    if not page.exists():
        return {"message": "The web page is not built yet. Try POST /ask."}
    return FileResponse(page)


def sources_for(passages, used_markers):
    out = []
    for marker in sorted(used_markers, key=lambda m: int(m[1:])):
        chunk = passages[int(marker[1:]) - 1]["chunk"]
        out.append({
            "marker": marker,
            "label": chunk["label"],
            "document": chunk["doc_title"],
            "date": chunk["version_date"],
            "url": chunk["source_url"],
            "excerpt": chunk["text"][:300],
        })
    return out


@app.post("/ask")
def ask(payload: Question):
    if retriever is None:
        raise HTTPException(503, "Still starting up, try again in a moment.")

    question = payload.question.strip()
    if question in cache:
        return {**cache[question], "cached": True}

    started = time.time()
    passages = retriever.search(question, n=PASSAGES_SHOWN, rerank=True)

    # Nothing good enough: say so rather than answering from a weak passage.
    if not passages or passages[0]["rerank_score"] < RERANK_MIN_SCORE:
        best = passages[0]["rerank_score"] if passages else None
        return {"answerable": False,
                "reason": "These documents do not appear to cover that question.",
                "claims": [], "sources": [],
                "best_score": best, "seconds": round(time.time() - started, 1)}

    complaint, problems = None, []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = ask_model(llm, question, passages, complaint)
        try:
            reply = parse_reply(raw)
        except json.JSONDecodeError:
            complaint, problems = "The reply was not valid JSON.", ["invalid JSON"]
            continue

        problems = validate(reply, passages)
        if problems:
            complaint = "\n".join(f"- {p}" for p in problems)
            continue

        if reply.get("answerable") is False:
            result = {"answerable": False,
                      "reason": reply.get("reason", "The passages do not answer this."),
                      "claims": [], "sources": [], "attempts": attempt}
        else:
            used = {str(c) for claim in reply["claims"] for c in claim["citations"]}
            result = {"answerable": True,
                      "claims": reply["claims"],
                      "sources": sources_for(passages, used),
                      "attempts": attempt}

        result["seconds"] = round(time.time() - started, 1)
        if len(cache) < CACHE_LIMIT:
            cache[question] = result
        return result

    # Both attempts failed the citation checks: refuse rather than show an
    # answer whose citations could not be verified.
    return {"answerable": False,
            "reason": "Could not produce a properly cited answer, so no answer is given.",
            "claims": [], "sources": [], "problems": problems,
            "seconds": round(time.time() - started, 1)}
