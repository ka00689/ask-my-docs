"""Search the documents and show what comes back. Replaces the earlier search.py.

Run it from the project folder with the virtual environment active:
    python src/search.py "From what date do high-risk obligations apply?"
    python src/search.py "Article 113" --mode keyword
    python src/search.py "Article 113" --compare
    python src/search.py "Is CV screening high-risk?" --rerank

Modes:
    hybrid  (default) keyword and vector results merged
    vector            meaning only, what you had before
    keyword           exact words only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import Retriever


def show(results, title):
    print("\n" + title)
    print("=" * 70)
    if not results:
        print("Nothing found.")
        return
    for rank, entry in enumerate(results, start=1):
        chunk, found_by = entry["chunk"], entry["found_by"]
        origin = ", ".join(f"{name} #{d['rank']}" for name, d in found_by.items())
        score = entry.get("rerank_score")
        grade = "" if score is None else f"   |  rerank score: {score:.2f}"
        print(f"\n{rank}. {chunk['label']}{grade}")
        print(f"   {chunk['doc_title']}  |  {chunk['version_date']}  |  found by: {origin}")
        print(f"   {chunk['source_url']}")
        print(f"   {chunk['text'][:300].replace(chr(10), ' ')}...")


def main():
    parser = argparse.ArgumentParser(description="Search the documents.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    parser.add_argument("--n", type=int, default=5, help="how many passages to show")
    parser.add_argument("--rerank", action="store_true",
                        help="have the cross-encoder pick the best passages")
    parser.add_argument("--decompose", action="store_true",
                        help="split a compound question into parts and search each")
    parser.add_argument("--compare", action="store_true",
                        help="show hybrid search before and after reranking")
    parser.add_argument("--exclude-recitals", action="store_true",
                        help="ignore recitals, keeping only articles and annexes")
    args = parser.parse_args()

    retriever = Retriever(exclude_kinds={"recital"} if args.exclude_recitals else ())
    print(f"\nQuestion: {args.question}")

    if args.compare:
        for mode in ("keyword", "vector", "hybrid"):
            show(retriever.search(args.question, mode=mode, n=args.n),
                 f"{mode.upper()} RESULTS")
        show(retriever.search(args.question, n=args.n, rerank=True),
             "HYBRID + RERANKED RESULTS")
        print("\n" + "=" * 70)
        print("Which list puts the passage that truly answers the question at the top?")
    else:
        title = f"{args.mode.upper()} RESULTS" + (" + RERANKED" if args.rerank else "")
        show(retriever.search(args.question, mode=args.mode, n=args.n,
                              rerank=args.rerank or args.decompose,
                              decompose=args.decompose), title)


if __name__ == "__main__":
    main()
