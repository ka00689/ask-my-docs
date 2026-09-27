"""The deployable interface: the same pipeline, as a Streamlit app.

Run it locally with:
    streamlit run streamlit_app.py

The retrieval, answering and citation checking are unchanged; only the page
layer differs. Streamlit builds the page from Python rather than serving the
HTML in web/, which is what lets it run on hosts that only support Python.
"""
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import streamlit as st

# Secrets set in the Community Cloud console arrive as st.secrets; copy them
# into the environment so llm.py finds them exactly as it does locally.
# Running locally there is no secrets file at all, and reading st.secrets then
# raises rather than returning empty, so the whole lookup is guarded: the key
# comes from .env in that case.
try:
    for key in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from answer import (ANSWER_SCHEMA, MAX_ATTEMPTS, PASSAGES_SHOWN, ask_model,
                    parse_reply, validate)
from llm import ModelChain
from retrieval import RERANK_MIN_SCORE, Retriever

QUESTIONS_PER_VISITOR = 5
SECONDS_BETWEEN_QUESTIONS = 5

EXAMPLES = [
    "Does the AI Act apply to open-source AI?",
    "What does the AI Act say about deep fake content?",
    "Can I be fined for how my AI system was trained?",
    "Which AI practices are prohibited?",
    "What is the AI literacy obligation?",
    "Is 2 August 2026 still the date when high-risk rules start?",
]

st.set_page_config(page_title="What does the AI Act say?", page_icon="📘",
                   layout="centered")


@st.cache_resource(show_spinner=False)
def load_system():
    """Loaded once for the whole app, not per visitor: the models are the

    largest thing in memory, and a free host does not have room for a copy
    per session.
    """
    index_dir = PROJECT / "data" / "index" / "chroma"
    if not index_dir.exists() or not any(index_dir.iterdir()):
        import subprocess
        with st.spinner("First run: building the search index. This takes a few minutes."):
            subprocess.run([sys.executable, str(PROJECT / "src" / "build_index.py")],
                           check=True)
    return Retriever(), ModelChain()


@st.cache_data(show_spinner=False, max_entries=500)
def answer_question(question):
    """Cached across everyone: a question asked twice costs one model call."""
    retriever, models = load_system()
    started = time.time()
    passages = retriever.search(question, n=PASSAGES_SHOWN, rerank=True)

    if not passages or passages[0]["rerank_score"] < RERANK_MIN_SCORE:
        return {"answerable": False,
                "reason": "These documents do not appear to cover that question.",
                "claims": [], "passages": [], "seconds": round(time.time() - started, 1)}

    complaint = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = ask_model(models, question, passages, complaint)
        try:
            reply = parse_reply(raw)
        except json.JSONDecodeError:
            complaint = "The reply was not valid JSON."
            continue

        problems = validate(reply, passages)
        if problems:
            complaint = "\n".join(f"- {p}" for p in problems)
            continue

        reply["passages"] = passages
        reply["attempts"] = attempt
        reply["seconds"] = round(time.time() - started, 1)
        return reply

    return {"answerable": False,
            "reason": "The model could not produce a properly cited answer, so none is shown.",
            "claims": [], "passages": [], "seconds": round(time.time() - started, 1)}


def within_limits():
    """Per-visitor limits, held in this browser session."""
    asked = st.session_state.setdefault("asked", deque())
    now = time.time()

    if asked and now - asked[-1] < SECONDS_BETWEEN_QUESTIONS:
        wait = int(SECONDS_BETWEEN_QUESTIONS - (now - asked[-1])) + 1
        return f"One question at a time, please. Try again in {wait} seconds."
    if len(asked) >= QUESTIONS_PER_VISITOR:
        return (f"You have asked {QUESTIONS_PER_VISITOR} questions, which is the "
                "limit for this demo. It runs on a free model allowance. "
                "Reload the page tomorrow, or try the example questions, which "
                "are answered from cache.")
    return None


def show(result):
    if not result.get("answerable"):
        st.warning(f"**No answer given.** {result.get('reason', '')}")
        return

    st.markdown("##### Answer")
    for claim in result["claims"]:
        markers = " ".join(f"`{c}`" for c in claim["citations"])
        st.markdown(f"{claim['text']} {markers}")

    used = sorted({str(c) for claim in result["claims"] for c in claim["citations"]},
                  key=lambda m: int(m[1:]))
    st.markdown("##### Sources")
    for marker in used:
        chunk = result["passages"][int(marker[1:]) - 1]["chunk"]
        st.markdown(
            f"**{marker}** · [{chunk['label']}]({chunk['source_url']})  \n"
            f"<span style='color:#6b7280'>{chunk['doc_title']} · version of "
            f"{chunk['version_date']}</span>", unsafe_allow_html=True)
        with st.expander("Show the passage this came from"):
            st.write(chunk["text"])

    note = [f"{result['seconds']}s"]
    if result.get("attempts", 1) > 1:
        note.append(f"{result['attempts']} drafts needed to pass the citation checks")
    st.caption(" · ".join(note))


st.title("What does the AI Act say?")
st.write("Ask a question about the EU's AI rules and get an answer built only from "
         "the legal text, with a link to the exact article it came from.")
st.info("It reads two documents: **Regulation (EU) 2024/1689**, the AI Act, and "
        "**Regulation (EU) 2026/1744**, the 2026 amendment that pushed back several "
        "deadlines. Ask about anything else and it will tell you it doesn't know, "
        "rather than making something up.")

with st.spinner("Loading the search index and models…"):
    load_system()

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Questions that work well:**")
columns = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if columns[i % 2].button(example, key=f"ex{i}", use_container_width=True):
        st.session_state.question = example

asked = st.text_area("Your question", value=st.session_state.question,
                     max_chars=500, height=80,
                     placeholder="What are the penalties for prohibited AI practices?")

if st.button("Ask", type="primary"):
    question = asked.strip()
    if len(question) < 3:
        st.warning("Please write a question first.")
    else:
        limited = within_limits()
        if limited:
            st.warning(limited)
        else:
            with st.spinner("Searching the regulation, then checking every citation…"):
                result = answer_question(question)
            st.session_state.asked.append(time.time())
            show(result)

st.divider()
st.caption(
    "Retrieval combines keyword and meaning-based search, a cross-encoder reranks "
    "the candidates, and a language model writes the answer. Every citation is "
    "checked in code before the answer is shown: claims citing a passage that was "
    "not provided, or stating a date absent from it, are rejected and rewritten. "
    "Not legal advice."
)
