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
from collections import deque
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from answer import (ANSWER_SCHEMA, MAX_ATTEMPTS, PASSAGES_SHOWN,
                    ask_model, parse_reply, validate)
from llm import ModelChain
from retrieval import RERANK_MIN_SCORE, Retriever

PROJECT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT / "web"

app = FastAPI(title="Ask My Docs", description="Question answering over the EU AI Act")

# Loaded once when the server starts, not per request: loading the models takes
# a few seconds and they can be reused for every question.
retriever = None
models = None
# Repeated questions are answered from memory. On a public demo most people ask
# the same few things, so this saves both time and, later, money.
cache = {}
CACHE_LIMIT = 500

# Limits, so one visitor (or a bot) cannot use up the day's free-tier quota and
# take the demo down for everyone else. Generous enough that nobody exploring
# the demo in good faith will notice.
QUESTIONS_PER_VISITOR_PER_DAY = int(os.environ.get("ASK_DAILY_LIMIT", "5"))
SECONDS_BETWEEN_QUESTIONS = int(os.environ.get("ASK_COOLDOWN", "5"))
QUESTIONS_PER_DAY_TOTAL = int(os.environ.get("ASK_TOTAL_DAILY_LIMIT", "200"))
DAY = 24 * 60 * 60

visits = {}          # visitor -> deque of timestamps
day_total = deque()  # timestamps of every answered question today


def visitor_of(request):
    """Identify a visitor well enough to rate-limit, without storing anything.

    Behind a host's proxy the client address is the proxy, so the forwarded
    header is used when present. Only kept in memory, only for counting, and
    lost whenever the server restarts.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_limits(visitor):
    """Returns a message to show the visitor, or None when they are within limits."""
    now = time.time()

    while day_total and now - day_total[0] > DAY:
        day_total.popleft()
    if len(day_total) >= QUESTIONS_PER_DAY_TOTAL:
        return ("This demo has answered its daily quota of questions. "
                "It runs on a free model allowance, which resets tomorrow.")

    seen = visits.setdefault(visitor, deque())
    while seen and now - seen[0] > DAY:
        seen.popleft()

    if seen and now - seen[-1] < SECONDS_BETWEEN_QUESTIONS:
        wait = int(SECONDS_BETWEEN_QUESTIONS - (now - seen[-1])) + 1
        return f"One question at a time, please. Try again in {wait} seconds."

    if len(seen) >= QUESTIONS_PER_VISITOR_PER_DAY:
        return (f"You have asked {QUESTIONS_PER_VISITOR_PER_DAY} questions today, "
                "which is the limit for this demo. The code is on GitHub if you "
                "would like to run your own copy without limits.")
    return None


def record_question(visitor):
    now = time.time()
    visits.setdefault(visitor, deque()).append(now)
    day_total.append(now)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def ensure_index():
    """Build the search index if this machine does not have one yet.

    data/index is not committed (it is rebuilt output, not source), so a freshly
    deployed copy has the chunks but no index. Building it takes a couple of
    minutes on first start and is then cached for as long as the host keeps the
    disk, which is why the health check reports readiness separately.
    """
    index_dir = PROJECT / "data" / "index" / "chroma"
    if index_dir.exists() and any(index_dir.iterdir()):
        return
    print("No search index found; building it from the chunks. This takes a few minutes...")
    import subprocess
    subprocess.run([sys.executable, str(PROJECT / "src" / "build_index.py")], check=True)
    print("Index built.")


@app.on_event("startup")
def load_everything():
    global retriever, models

    ensure_index()
    print("Loading retriever (this takes a few seconds)...")
    retriever = Retriever()
    models = ModelChain()
    print(f"Answering with: {', '.join(str(p) for p in models.providers)}")
    print(f"Ready: {len(retriever.chunks)} chunks indexed.")


@app.get("/health")
def health():
    return {"status": "ok",
            "questions_today": len(day_total),
            "daily_capacity": QUESTIONS_PER_DAY_TOTAL,
            "chunks": len(retriever.chunks) if retriever else 0,
            "model": models.primary.model if models else None}


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
def ask(payload: Question, request: Request):
    if retriever is None:
        raise HTTPException(503, "Still starting up, try again in a moment.")

    question = payload.question.strip()

    # A cached answer costs nothing, so it is served before the limits apply.
    if question in cache:
        return {**cache[question], "cached": True}

    visitor = visitor_of(request)
    limited = check_limits(visitor)
    if limited:
        return {"answerable": False, "reason": limited, "claims": [],
                "sources": [], "rate_limited": True}
    record_question(visitor)

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
        raw = ask_model(models, question, passages, complaint)
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
