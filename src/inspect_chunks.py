"""Inspect the chunks produced by extract_chunks.py, in a readable way.

Run it from the project folder with the virtual environment active:
    python src/inspect_chunks.py            # automatic checks + a few samples
    python src/inspect_chunks.py art_113    # show every chunk whose id matches
"""
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT / "data" / "processed" / "chunks.jsonl"

# Words that would suggest web-page clutter survived the cleaning step.
CLUTTER = ["cookie", "skip to main content", "javascript", "navigation menu",
           "©", "print this page"]


def load():
    if not CHUNKS_FILE.exists():
        sys.exit(f"{CHUNKS_FILE} not found. Run: python src/extract_chunks.py")
    return [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]


def show(chunk):
    print("-" * 70)
    print(f"id     : {chunk['chunk_id']}")
    print(f"label  : {chunk['label']}")
    where = " / ".join(x for x in (chunk["chapter"], chunk["section"]) if x)
    print(f"where  : {where or '(no chapter)'}")
    print(f"part   : {chunk['part']} of {chunk['of_parts']}   length: {len(chunk['text'])} characters")
    print(f"link   : {chunk['source_url']}")
    print("text   :")
    text = chunk["text"]
    print("   " + (text if len(text) <= 700 else text[:350] + "\n   [...]\n   " + text[-350:]))


def run_checks(chunks):
    print("\n" + "=" * 70)
    print("AUTOMATIC CHECKS")
    print("=" * 70)

    # 1. Does anything look like leftover web-page furniture?
    dirty = [c for c in chunks
             if any(word in c["text"].lower() for word in CLUTTER)]
    print(f"Chunks containing web-page clutter: {len(dirty)}  (want 0)")
    for c in dirty[:3]:
        print(f"   - {c['chunk_id']}")

    # 2. Do chunks end at a sensible place?
    mid_sentence = [c for c in chunks
                    if c["part"] == c["of_parts"] and not re.search(r"[.;:)\]]$", c["text"].strip())]
    print(f"Final chunks not ending in punctuation: {len(mid_sentence)}  (a few is fine)")
    for c in mid_sentence[:3]:
        print(f"   - {c['chunk_id']}: ...{c['text'][-60:]}")

    # 3. Does the label agree with the text?
    mismatched = []
    for c in chunks:
        m = re.match(r"Article (\d+)", c["label"])
        if m and c["part"] == 1 and f"Article {m.group(1)} " not in c["text"][:120]:
            mismatched.append(c)
    print(f"Article chunks whose text doesn't start with their own number: {len(mismatched)}  (want 0)")
    for c in mismatched[:3]:
        print(f"   - {c['chunk_id']} labelled {c['label'][:40]}")

    # 4. Are any articles missing entirely?
    for doc_id in sorted({c["doc_id"] for c in chunks}):
        numbers = {int(m.group(1)) for c in chunks if c["doc_id"] == doc_id
                   for m in [re.match(r"Article (\d+)", c["label"])] if m}
        if numbers:
            missing = sorted(set(range(1, max(numbers) + 1)) - numbers)
            print(f"{doc_id}: articles 1-{max(numbers)} present, missing: "
                  f"{missing if missing else 'none'}")

    # 5. Very short chunks are usually a sign of a parsing problem.
    tiny = sorted((c for c in chunks if len(c["text"]) < 200), key=lambda c: len(c["text"]))
    print(f"Chunks shorter than 200 characters: {len(tiny)}  (a few short recitals are normal)")
    for c in tiny[:3]:
        print(f"   - {c['chunk_id']} ({len(c['text'])}): {c['text'][:80]}")


def main():
    chunks = load()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.relative_to(PROJECT)}")

    if len(sys.argv) > 1:
        needle = sys.argv[1].lower()
        matches = [c for c in chunks if needle in c["chunk_id"].lower()
                   or needle in c["label"].lower()]
        print(f"{len(matches)} chunk(s) matching '{sys.argv[1]}'")
        for c in matches[:10]:
            show(c)
        return

    print("\nA SAMPLE FROM ACROSS THE FILE (first, quarter, middle, three-quarters, last):")
    n = len(chunks)
    for i in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        show(chunks[i])

    run_checks(chunks)
    print("\nTip: to look at one article, run:  python src/inspect_chunks.py art_113")


if __name__ == "__main__":
    main()
