"""Retrieval: keyword search (BM25), vector search (Chroma), and the two merged.

This module is imported by search.py and, later, by the answering code.
It is not meant to be run directly.

Why both:
  - Vector search understands meaning ("when do the rules bite" -> "shall apply from"),
    but it treats "Article 113" as just another topic word.
  - BM25 matches exact words, so article numbers, dates and defined terms land
    where they should, but it misses paraphrases.
Merging the two lists gives you the strengths of each.
"""
import json
import os
import re
from pathlib import Path

# Privacy: switch off anonymous usage statistics.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT / "data" / "index" / "chroma"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "ai_act"

# How many candidates each method contributes before merging.
CANDIDATES_PER_METHOD = 20
# Reciprocal rank fusion constant. 60 is the usual starting value; it stops the
# very top result of one method from dominating the merged list.
RRF_K = 60

# How many merged candidates the reranker reads before choosing the best few.
RERANK_CANDIDATES = 20
# The reranker scores each passage; below this, a passage is probably not an
# answer at all. Used later to let the system say "I don't know".
RERANK_MIN_SCORE = 0.0

# Recitals are the explanatory preamble of an EU regulation. They read like
# answers, so they crowd out the articles that actually contain the law. Rather
# than dropping them (some questions can only be answered from a recital), they
# are demoted: their fusion vote counts for less, and the reranker's score for
# them is reduced. Both values are tunable and were chosen by measurement.
RECITAL_VOTE_WEIGHT = 0.0     # 1.0 = no demotion, 0.0 = recitals never merge in
RECITAL_SCORE_PENALTY = 3.0   # subtracted from a recital's rerank score

# Compound questions ("penalties for prohibited practices") need two unrelated
# articles: one defines the practices, another sets the penalties. Neither
# passage matches the whole question well, so the question is split and each
# part searched separately. This costs one call to the local model per question.
DECOMPOSE_MODEL = "llama3.2"
MAX_SUBQUESTIONS = 3

# Long articles were split into parts (art_99_p1, art_99_p2). A question can
# match one part while the answer sits in the next, so when a part reaches the
# shortlist its siblings are added for the reranker to judge as well.
INCLUDE_SIBLING_PARTS = True


def tokenize(text):
    """Split text into lowercase words and numbers, so 'Article 113' -> ['article', '113']."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, exclude_kinds=(),
                 recital_vote_weight=RECITAL_VOTE_WEIGHT,
                 recital_score_penalty=RECITAL_SCORE_PENALTY,
                 include_sibling_parts=INCLUDE_SIBLING_PARTS):
        # e.g. exclude_kinds={"recital"} to see what recitals are costing you.
        self.exclude_kinds = set(exclude_kinds)
        self.recital_vote_weight = recital_vote_weight
        self.recital_score_penalty = recital_score_penalty
        self.include_sibling_parts = include_sibling_parts
        if not CHUNKS_FILE.exists():
            raise SystemExit(f"{CHUNKS_FILE} not found. Run: python src/extract_chunks.py")
        if not INDEX_DIR.exists():
            raise SystemExit(f"No index at {INDEX_DIR}. Run: python src/build_index.py")

        self.chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        # Keyword index: built in memory each run. With a few hundred chunks
        # this takes a fraction of a second.
        self.bm25 = BM25Okapi([tokenize(f"{c['label']} {c['text']}") for c in self.chunks])

        # Vector index: already built and saved by build_index.py.
        # Group the parts of each split article, e.g. art_99_p1 and art_99_p2.
        self.siblings = {}
        for c in self.chunks:
            base = re.sub(r"_p\d+$", "", c["chunk_id"])
            self.siblings.setdefault(base, []).append(c["chunk_id"])

        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # The reranker is slower to load, so only load it if it gets used.
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def allowed(self, chunk_id):
        return self.by_id[chunk_id]["kind"] not in self.exclude_kinds

    def keyword_search(self, question, n=CANDIDATES_PER_METHOD):
        """Best matches by exact words. Returns [(chunk_id, score), ...]."""
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [(self.chunks[i]["chunk_id"], float(scores[i])) for i in ranked if scores[i] > 0]
        return [r for r in out if self.allowed(r[0])][:n]

    def vector_search(self, question, n=CANDIDATES_PER_METHOD):
        """Best matches by meaning. Returns [(chunk_id, score), ...]."""
        embedding = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        # Fetch extra when filtering, so the shortlist is still full afterwards.
        fetch = n * 3 if self.exclude_kinds else n
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(fetch, self.collection.count()),
            include=["distances"],
        )
        pairs = [(cid, 1 - dist)
                 for cid, dist in zip(results["ids"][0], results["distances"][0])]
        return [p for p in pairs if self.allowed(p[0])][:n]

    def hybrid_search(self, question, n=10):
        """Both methods, merged with reciprocal rank fusion.

        Each method votes for a chunk with a weight of 1/(60 + its rank), and the
        votes are added up. A chunk found by both methods therefore beats one
        found by only a single method, without either method's raw scores
        (which are on completely different scales) having to be compared.
        """
        keyword = self.keyword_search(question)
        vector = self.vector_search(question)

        fused = {}
        for method_name, results in (("keyword", keyword), ("vector", vector)):
            for rank, (chunk_id, score) in enumerate(results, start=1):
                entry = fused.setdefault(
                    chunk_id, {"chunk_id": chunk_id, "rrf": 0.0, "found_by": {}}
                )
                weight = (self.recital_vote_weight
                          if self.by_id[chunk_id]["kind"] == "recital" else 1.0)
                entry["rrf"] += weight * (1 / (RRF_K + rank))
                entry["found_by"][method_name] = {"rank": rank, "score": score}

        merged = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:n]
        for entry in merged:
            entry["chunk"] = self.by_id[entry["chunk_id"]]
        return merged

    def decompose(self, question):
        """Split a compound question into parts. Returns [question] if it is simple."""
        from langchain_ollama import ChatOllama

        schema = {
            "type": "object",
            "properties": {
                "compound": {"type": "boolean"},
                "subquestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["compound", "subquestions"],
        }
        instruction = (
            "Decide whether this question needs information from two or more "
            "separate parts of a law. If it does, set compound true and list "
            f"up to {MAX_SUBQUESTIONS} standalone sub-questions, each answerable "
            "on its own. If one passage could answer it, set compound false and "
            "return an empty list. Reply with JSON only.\n\n"
            "Example: 'What are the penalties for prohibited AI practices?' is "
            "compound, because one part of the law defines prohibited practices "
            "and another sets penalties.\n\n"
            f"QUESTION: {question}"
        )
        try:
            llm = ChatOllama(model=DECOMPOSE_MODEL, temperature=0, format=schema)
            reply = json.loads(llm.invoke(instruction).content)
        except Exception:
            return [question]      # if the model is unavailable, carry on as normal

        parts = [q.strip() for q in reply.get("subquestions", []) if q and q.strip()]
        if not reply.get("compound") or not parts:
            return [question]
        # Keep the original too: it carries the full intent.
        return [question] + parts[:MAX_SUBQUESTIONS]

    def multi_search(self, questions, n=RERANK_CANDIDATES):
        """Run hybrid search for each question and merge the results."""
        fused = {}
        for q in questions:
            for rank, entry in enumerate(self.hybrid_search(q, n=n), start=1):
                merged = fused.setdefault(entry["chunk_id"], {
                    "chunk_id": entry["chunk_id"], "rrf": 0.0,
                    "found_by": {}, "chunk": entry["chunk"],
                })
                merged["rrf"] += 1 / (RRF_K + rank)
                merged["found_by"].update(entry["found_by"])
        return sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:n]

    def add_sibling_parts(self, entries, limit=10):
        """Add the other parts of any split article already on the shortlist."""
        present = {e["chunk_id"] for e in entries}
        added = []
        for entry in list(entries):
            base = re.sub(r"_p\d+$", "", entry["chunk_id"])
            for sibling in self.siblings.get(base, []):
                if sibling not in present and self.allowed(sibling) and len(added) < limit:
                    present.add(sibling)
                    added.append({"chunk_id": sibling, "rrf": 0.0,
                                  "found_by": {"sibling": {"rank": 0, "score": 0.0}},
                                  "chunk": self.by_id[sibling]})
        return entries + added

    def rerank(self, question, entries, n=5):
        """Read each candidate against the question and keep the best ones.

        Search finds passages that look related. The reranker instead reads the
        question and the passage together, the way a person would, and scores how
        well that passage actually answers it. It is slower, which is why it only
        sees the shortlist rather than all the chunks.
        """
        if not entries:
            return []
        pairs = [(question, f"{e['chunk']['label']}\n{e['chunk']['text']}") for e in entries]
        scores = self.reranker.predict(pairs)
        for entry, score in zip(entries, scores):
            entry["rerank_score_raw"] = float(score)
            penalty = (self.recital_score_penalty
                       if entry["chunk"]["kind"] == "recital" else 0.0)
            entry["rerank_score"] = float(score) - penalty
        ranked = sorted(entries, key=lambda e: e["rerank_score"], reverse=True)
        return ranked[:n]

    def search(self, question, mode="hybrid", n=10, rerank=False, decompose=False):
        """One entry point for all modes, so they are easy to compare."""
        if rerank:
            if decompose:
                candidates = self.multi_search(self.decompose(question))
            else:
                candidates = self.hybrid_search(question, n=RERANK_CANDIDATES)
            if self.include_sibling_parts:
                candidates = self.add_sibling_parts(candidates)
            return self.rerank(question, candidates, n=n)

        if mode == "hybrid":
            return self.hybrid_search(question, n=n)

        raw = (self.keyword_search(question, n=n) if mode == "keyword"
               else self.vector_search(question, n=n))
        return [{"chunk_id": cid,
                 "rrf": None,
                 "found_by": {mode: {"rank": rank, "score": score}},
                 "chunk": self.by_id[cid]}
                for rank, (cid, score) in enumerate(raw, start=1)]
