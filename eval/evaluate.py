"""Measure how well retrieval finds the right passages.

Run it from the project folder with the virtual environment active:
    python eval/evaluate.py                    # all modes, k=5
    python eval/evaluate.py --k 10
    python eval/evaluate.py --mode rerank --show-failures

It reads eval/questions.jsonl, where each question lists the chunk ids that
should be found, and reports:

  hit rate   how often at least one correct passage is in the top k
  recall     what share of the correct passages are in the top k
  MRR        how high the first correct passage ranks (1.0 = always first)
  refusals   for questions with no answer in the documents, how often the best
             passage scores below the threshold, i.e. the system would say
             "I don't know" instead of inventing something

No AI model writes answers here, so this is free, fast and gives the same
result every time. That is what makes it safe to run automatically on every
change later.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from retrieval import RERANK_MIN_SCORE, Retriever

QUESTIONS_FILE = PROJECT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT / "eval" / "results"

MODES = {
    "vector": dict(mode="vector", rerank=False),
    "keyword": dict(mode="keyword", rerank=False),
    "hybrid": dict(mode="hybrid", rerank=False),
    "rerank": dict(mode="hybrid", rerank=True),
    "decompose": dict(mode="hybrid", rerank=True, decompose=True),
}


def load_questions(retriever):
    if not QUESTIONS_FILE.exists():
        sys.exit(f"{QUESTIONS_FILE} not found.")
    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")
                 if line.strip()]

    # A test set that points at chunks which no longer exist would quietly
    # report failures forever, so check the ids first.
    unknown = [(q["id"], g) for q in questions for g in q["gold_chunk_ids"]
               if g not in retriever.by_id]
    if unknown:
        print("WARNING: these questions point at chunk ids that are not in your index:")
        for qid, gold in unknown:
            print(f"   {qid}: {gold}")
        print("Fix the ids (or rebuild the index) before trusting these numbers.\n")
    return questions


def evaluate_mode(retriever, questions, mode_name, k):
    settings = MODES[mode_name]
    answerable = [q for q in questions if q["gold_chunk_ids"]]
    unanswerable = [q for q in questions if not q["gold_chunk_ids"]]

    hits = recalls = reciprocal_ranks = 0.0
    per_question = []

    for q in answerable:
        results = retriever.search(q["question"], n=k, **settings)
        found = [entry["chunk_id"] for entry in results]
        gold = set(q["gold_chunk_ids"])

        matched = [i for i, cid in enumerate(found, start=1) if cid in gold]
        hit = bool(matched)
        recall = len(gold & set(found)) / len(gold)
        rr = 1 / matched[0] if matched else 0.0

        hits += hit
        recalls += recall
        reciprocal_ranks += rr
        per_question.append({"id": q["id"], "question": q["question"], "tags": q["tags"],
                             "hit": hit, "recall": recall, "rr": rr,
                             "retrieved": found[:k], "gold": q["gold_chunk_ids"]})

    # Would the system correctly refuse the questions it cannot answer?
    correct_refusals = 0
    for q in unanswerable:
        results = retriever.search(q["question"], n=k, mode="hybrid", rerank=True)
        best = results[0].get("rerank_score", 0.0) if results else -99
        refused = best < RERANK_MIN_SCORE
        correct_refusals += refused
        per_question.append({"id": q["id"], "question": q["question"], "tags": q["tags"],
                             "hit": refused, "recall": float(refused), "rr": float(refused),
                             "best_score": best, "gold": []})

    n = len(answerable) or 1
    return {
        "mode": mode_name,
        "k": k,
        "questions": len(questions),
        "hit_rate": hits / n,
        "recall": recalls / n,
        "mrr": reciprocal_ranks / n,
        "refusal_rate": correct_refusals / len(unanswerable) if unanswerable else None,
        "per_question": per_question,
    }


def by_tag(result):
    """Average hit rate for each tag, to show where the system struggles."""
    tags = {}
    for q in result["per_question"]:
        for tag in q["tags"]:
            tags.setdefault(tag, []).append(q["hit"])
    return {tag: sum(v) / len(v) for tag, v in sorted(tags.items())}


def main():
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--k", type=int, default=5, help="how many passages count as 'found'")
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    parser.add_argument("--show-failures", action="store_true",
                        help="list the questions where nothing correct was found")
    parser.add_argument("--save", action="store_true", help="write results to eval/results/")
    parser.add_argument("--min-hit-rate", type=float, default=None,
                        help="fail (exit 1) if hit rate falls below this, e.g. 0.80")
    parser.add_argument("--min-mrr", type=float, default=None,
                        help="fail (exit 1) if MRR falls below this, e.g. 0.65")
    parser.add_argument("--min-refusal-rate", type=float, default=None,
                        help="fail (exit 1) if correct refusals fall below this")
    parser.add_argument("--exclude-recitals", action="store_true",
                        help="experiment: ignore recitals, keeping only articles and annexes")
    parser.add_argument("--recital-weight", type=float, default=None,
                        help="how much a recital's fusion vote counts (1.0 = no demotion)")
    parser.add_argument("--recital-penalty", type=float, default=None,
                        help="how much to subtract from a recital's rerank score")
    parser.add_argument("--no-siblings", action="store_true",
                        help="turn off adding the other parts of a split article")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="try several refusal thresholds and report the trade-off")
    parser.add_argument("--sweep", action="store_true",
                        help="try several demotion settings and report which is best")
    args = parser.parse_args()

    kwargs = {}
    if args.recital_weight is not None:
        kwargs["recital_vote_weight"] = args.recital_weight
    if args.recital_penalty is not None:
        kwargs["recital_score_penalty"] = args.recital_penalty
    if args.no_siblings:
        kwargs["include_sibling_parts"] = False

    retriever = Retriever(exclude_kinds={"recital"} if args.exclude_recitals else (), **kwargs)
    questions = load_questions(retriever)

    if args.sweep_threshold:
        import retrieval as R
        answerable = [q for q in questions if q["gold_chunk_ids"]]
        unanswerable = [q for q in questions if not q["gold_chunk_ids"]]
        print(f"\nRefusal threshold trade-off, {len(questions)} questions, k={args.k}.")
        print("answered = share of answerable questions the system would attempt.")
        print("refused  = share of out-of-scope questions it correctly declines.\n")
        header = f"{'threshold':>10} {'answered':>9} {'refused':>9}"
        print(header); print("-" * len(header))

        scored = []      # best rerank score per question, computed once
        for q in questions:
            results = retriever.search(q["question"], n=args.k, mode="hybrid", rerank=True)
            best = results[0]["rerank_score"] if results else -99
            scored.append((q, best))

        for threshold in (0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0):
            attempted = sum(1 for q, b in scored if q["gold_chunk_ids"] and b >= threshold)
            declined = sum(1 for q, b in scored if not q["gold_chunk_ids"] and b < threshold)
            print(f"{threshold:>10.1f} {attempted / max(len(answerable), 1):>8.0%} "
                  f"{declined / max(len(unanswerable), 1):>8.0%}")
        print("\nPick the loosest threshold that still refuses every out-of-scope question.")
        return

    if args.sweep:
        print(f"\nTuning recital demotion on {len(questions)} questions, k={args.k}.")
        print("Each row demotes recitals a bit more. Higher hit rate and MRR is better.\n")
        header = f"{'vote weight':>11} {'penalty':>8} {'hit rate':>9} {'recall':>8} {'MRR':>7}"
        print(header)
        print("-" * len(header))
        for weight, penalty in [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (0.5, 1.0),
                                (0.5, 2.0), (0.25, 1.0), (0.0, 3.0)]:
            r_ = Retriever(recital_vote_weight=weight, recital_score_penalty=penalty)
            res = evaluate_mode(r_, questions, "rerank", args.k)
            print(f"{weight:>11.2f} {penalty:>8.1f} {res['hit_rate']:>8.0%} "
                  f"{res['recall']:>8.0%} {res['mrr']:>7.2f}")
        print("\nPick the row that keeps recital-only questions working "
              "(q21) while ranking articles first.")
        return
    if args.exclude_recitals:
        print("\nEXPERIMENT: recitals excluded from retrieval.")
        skipped = [q["id"] for q in questions
                   if q["gold_chunk_ids"] and all("rct_" in g for g in q["gold_chunk_ids"])]
        if skipped:
            print(f"Note: {len(skipped)} question(s) can now never be answered "
                  f"because their only correct passages are recitals: {', '.join(skipped)}")
    modes = list(MODES) if args.mode == "all" else [args.mode]

    print(f"\n{len(questions)} questions, top {args.k} passages count as found.\n")
    header = f"{'mode':10} {'hit rate':>9} {'recall':>8} {'MRR':>7} {'refusals':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for mode_name in modes:
        r = evaluate_mode(retriever, questions, mode_name, args.k)
        results.append(r)
        refusal = "-" if r["refusal_rate"] is None else f"{r['refusal_rate']:.0%}"
        print(f"{mode_name:10} {r['hit_rate']:>8.0%} {r['recall']:>8.0%} "
              f"{r['mrr']:>7.2f} {refusal:>9}")

    best = results[-1]
    print(f"\nBy question type ({best['mode']} mode):")
    for tag, score in by_tag(best).items():
        print(f"   {tag:14} {score:.0%}")

    if args.show_failures:
        print(f"\nQuestions where {best['mode']} found nothing correct:")
        for q in best["per_question"]:
            if not q["hit"]:
                print(f"\n   [{q['id']}] {q['question']}")
                print(f"      wanted:    {q['gold'] or 'a refusal'}")
                print(f"      retrieved: {q.get('retrieved', [])[:3]}")

    # The gate: turn the measurement into a pass or fail, so CI can block a
    # change that makes retrieval worse.
    thresholds = [("hit rate", best["hit_rate"], args.min_hit_rate),
                  ("MRR", best["mrr"], args.min_mrr),
                  ("refusal rate", best["refusal_rate"], args.min_refusal_rate)]
    checked = [(name, got, want) for name, got, want in thresholds if want is not None]
    if checked:
        print("\nTHRESHOLDS")
        print("-" * 40)
        failures = []
        for name, got, want in checked:
            got = -1.0 if got is None else got
            ok = got >= want
            print(f"   {name:14} {got:.2f}  needs >= {want:.2f}   {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{name} {got:.2f} < {want:.2f}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{date.today().isoformat()}_k{args.k}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to {out.relative_to(PROJECT)}")

    if checked and failures:
        print("\nGATE FAILED: " + "; ".join(failures))
        print("This change makes retrieval worse. Fix it or justify moving the threshold.")
        sys.exit(1)
    if checked:
        print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
