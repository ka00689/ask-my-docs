"""Answer a question from the retrieved passages, with citations your code checks.

Run it from the project folder with the virtual environment active:
    python src/answer.py "From what date do high-risk obligations apply?"
    python src/answer.py "What is the capital of France?"     # should refuse

How the enforcement works:
  1. Retrieve the best passages and number them [1]...[5].
  2. Ask the local model for a JSON answer where every claim lists the passage
     numbers that support it.
  3. Check the reply in code: valid JSON, every claim cited, every citation a
     real passage number, and every date or figure in a claim actually present
     in the passages it cites.
  4. If a check fails, ask once more with the specific problem named. If it
     fails again, refuse rather than show an unverified answer.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import ModelChain
from retrieval import RERANK_MIN_SCORE, Retriever

MODEL = "qwen2.5:7b"   # llama3.2 could not follow the citation rules reliably
# How many passages the model is shown. Tried 8; measured worse, because the
# extra low-scoring passages gave the model more ways to go wrong. Kept at 5.
PASSAGES_SHOWN = 5
# Small models often get the citation rules right on a later try, so give them
# a few goes before refusing. Each attempt names the specific problem found.
MAX_ATTEMPTS = 4

# Ollama can force the reply to match this shape while it generates, so a claim
# with no citation becomes impossible rather than something we catch afterwards.
# "minItems": 1 is what stops the empty citation lists you were getting.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reason": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^S[0-9]+$"},
                        "minItems": 1,
                    },
                },
                "required": ["text", "citations"],
            },
        },
    },
    "required": ["answerable", "claims"],
}

SYSTEM_PROMPT = """You answer questions about EU law using ONLY the numbered passages provided.

Rules:
- Use only what the passages say. Never add knowledge from memory.
- Be brief. Answer in at most three short claims, in your own words.
  Summarise; do not copy long stretches of the passage.
- Answer the question that was asked. If a passage lists several dates,
  categories or cases, give only the one the question is about, and say which
  one it is. Never give a date for a different category.
- Every claim must cite the passage markers it comes from, written exactly as shown: S1, S2, S3.
- Cite the marker at the top of a passage, NEVER an article or recital number from inside the text.
  A passage headed [S2] about "Recital 40" is cited as "S2", not "40".
- If the passages do not answer the question, say so with "answerable": false.
- When passages disagree, prefer the one with the later version_date and say that it amends the earlier one.
- Regulation (EU) 2024/1689 was amended in 2026 by Regulation (EU) 2026/1744,
  which changed several dates of application. If the question is about dates,
  deadlines or when obligations start, and no passage from the 2026 amendment
  is among those provided, add a final claim warning that the dates shown are
  from the original 2024 text and may have been changed by the 2026 amendment,
  citing the passage the dates came from.

Reply with JSON only, in exactly this shape:
{"answerable": true, "claims": [{"text": "A sentence of the answer.", "citations": ["S1", "S3"]}]}
or
{"answerable": false, "reason": "Why the passages do not answer it.", "claims": []}

"citations" must never be empty. Every claim needs at least one passage number.

Worked example. Given passages:
[S1] Article 99 - Penalties
Source: Example Regulation (version_date: 2024-07-12, binding_law)
Article 99 Penalties 1. Non-compliance with the prohibitions shall be subject to
administrative fines of up to EUR 35 000 000.
[S2] Recital 5
Source: Example Regulation (version_date: 2024-07-12, binding_law)
(5) Penalties should be effective, proportionate and dissuasive.

Question: What is the maximum fine for prohibited practices?

Correct reply:
{"answerable": true, "claims": [{"text": "Prohibited practices can be fined up to EUR 35 000 000.", "citations": ["S1"]}, {"text": "Penalties are required to be effective, proportionate and dissuasive.", "citations": ["S2"]}]}

Note how "Recital 5" is cited as "S2", the marker, not as 5."""


def build_prompt(question, passages):
    """Lay out the numbered passages followed by the question."""
    blocks = []
    for i, entry in enumerate(passages, start=1):
        c = entry["chunk"]
        blocks.append(
            f"[S{i}] {c['label']}\n"
            f"Source: {c['doc_title']} (version_date: {c['version_date']}, {c['authority']})\n"
            f"{c['text']}"
        )
    return f"PASSAGES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


def parse_reply(raw):
    """Turn the model's text into a dict, tolerating code fences."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


MONTHS = {"january": "01", "february": "02", "march": "03", "april": "04",
          "may": "05", "june": "06", "july": "07", "august": "08",
          "september": "09", "october": "10", "november": "11", "december": "12"}


def normalise_dates(text):
    """Write dates one way, so "12 July 2024" and "July 12, 2024" compare equal."""
    text = re.sub(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
                  lambda m: f"{m.group(3)}-{MONTHS.get(m.group(2).lower(), m.group(2))}-{int(m.group(1)):02d}",
                  text)
    text = re.sub(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b",
                  lambda m: f"{m.group(3)}-{MONTHS.get(m.group(1).lower(), m.group(1))}-{int(m.group(2)):02d}",
                  text)
    return text


# Dates and figures are what a model is most likely to invent, so they get checked.
FACT_PATTERN = re.compile(
    r"\b(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}/\d{1,4}|\d+(?:\.\d+)?\s*%|EUR\s?[\d\s,.]+\d)\b"
)


def validate(reply, passages):
    """Return a list of problems. An empty list means the answer passed."""
    problems = []

    if not isinstance(reply, dict) or "answerable" not in reply:
        return ["Reply is not in the required shape."]

    if reply.get("answerable") is False:
        return []      # a refusal needs no citations

    claims = reply.get("claims") or []
    if not claims:
        return ["Said the question was answerable but gave no claims."]

    valid_markers = {f"S{i}" for i in range(1, len(passages) + 1)}

    for i, claim in enumerate(claims, start=1):
        text = (claim.get("text") or "").strip()
        citations = claim.get("citations") or []

        if not text:
            problems.append(f"Claim {i} has no text.")
            continue
        if not citations:
            problems.append(f"Claim {i} has no citation: {text[:60]}")
            continue

        bad = [c for c in citations if str(c) not in valid_markers]
        if bad:
            problems.append(
                f"Claim {i} cites {bad}, which is not a passage marker. "
                f"Use only {sorted(valid_markers)} - the marker at the top of a passage, "
                f"not an article or recital number from inside it."
            )
            continue

        # Every date or figure in the claim must appear in a cited passage.
        # Only the passage's own text counts: the "Source: ... version_date: ..."
        # header we add around it is our furniture, not the law, and a model
        # quoting a date from there would otherwise pass this check.
        cited_text = " ".join(passages[int(str(c)[1:]) - 1]["chunk"]["text"] for c in citations)
        cited_text = normalise_dates(cited_text)
        for fact in FACT_PATTERN.findall(text):
            if normalise_dates(fact) not in cited_text:
                problems.append(
                    f"Claim {i} states '{fact}' but that does not appear in the cited passage(s)."
                )

    return problems


def ask_model(models, question, passages, complaint=None):
    prompt = build_prompt(question, passages)
    if complaint:
        prompt += ("\n\nYour previous reply was rejected for these reasons:\n"
                   f"{complaint}\n"
                   "Write it again. Copy every date, figure and regulation number "
                   "exactly as it appears in the passage you cite, and if a fact "
                   "is not in the passages, leave it out.")
    text, _provider = models.complete(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA)
    return text


def print_answer(reply, passages):
    if reply.get("answerable") is False:
        print("\nNo answer given.")
        print(f"Reason: {reply.get('reason', 'the passages do not cover this question')}")
        return

    print("\nANSWER")
    print("=" * 70)
    for claim in reply["claims"]:
        marks = "".join(f"[{c}]" for c in claim["citations"])
        print(f"{claim['text']} {marks}")

    used = sorted({str(c) for claim in reply["claims"] for c in claim["citations"]},
                  key=lambda m: int(m[1:]))
    print("\nSOURCES")
    print("=" * 70)
    for marker in used:
        chunk = passages[int(marker[1:]) - 1]["chunk"]
        print(f"[{marker}] {chunk['label']} - {chunk['doc_title']} ({chunk['version_date']})")
        print(f"    {chunk['source_url']}")


def main():
    parser = argparse.ArgumentParser(description="Answer a question with checked citations.")
    parser.add_argument("question")
    parser.add_argument("--show-passages", action="store_true",
                        help="also print the passages the model was given")
    parser.add_argument("--debug", action="store_true",
                        help="print the model's raw reply, before any checking")
    parser.add_argument("--model", default=MODEL,
                        help="which Ollama model to use, e.g. qwen2.5:7b")
    args = parser.parse_args()

    retriever = Retriever()
    passages = retriever.search(args.question, n=PASSAGES_SHOWN, rerank=True)

    # If even the best passage scores poorly, the documents probably do not cover this.
    if not passages or passages[0]["rerank_score"] < RERANK_MIN_SCORE:
        best = f"{passages[0]['rerank_score']:.2f}" if passages else "none"
        print(f"\nNo answer given. Nothing relevant found (best score: {best}).")
        print("These documents cover the EU AI Act and its 2026 amendment.")
        return

    if args.show_passages:
        print("\nPASSAGES GIVEN TO THE MODEL")
        print("=" * 70)
        for i, entry in enumerate(passages, start=1):
            print(f"[S{i}] {entry['chunk']['label']}  (score {entry['rerank_score']:.2f})")

    models = ModelChain(ollama_model=args.model)
    if args.debug:
        print(f"(model providers in order: {', '.join(str(p) for p in models.providers)})")

    complaint = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = ask_model(models, args.question, passages, complaint)
        if args.debug:
            print(f"\n--- raw reply, attempt {attempt} ---\n{raw}\n--- end ---")
        try:
            reply = parse_reply(raw)
        except json.JSONDecodeError:
            complaint = "The reply was not valid JSON."
            print(f"(attempt {attempt}: reply was not valid JSON, retrying)")
            continue

        problems = validate(reply, passages)
        if not problems:
            print_answer(reply, passages)
            return

        complaint = "\n".join(f"- {p}" for p in problems)
        print(f"(attempt {attempt} rejected:)")
        for p in problems:
            print(f"   - {p}")

    print("\nNo answer given. The model could not produce a properly cited answer.")
    print("Refusing is the right outcome here: an unverified answer is worse than none.")


if __name__ == "__main__":
    main()
