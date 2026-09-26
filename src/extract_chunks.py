"""Step 1 of ingestion: turn EUR-Lex HTML into clean, labelled chunks.

Run it from the project folder with the virtual environment active:
    python src/extract_chunks.py

It reads every .html file in data/raw/law, splits each law along its own
structure (one chunk per article, recital or annex), and writes the result to
data/processed/chunks.jsonl so you can inspect it before anything is embedded.
"""
import json
import os
import re
import sys
from pathlib import Path

# Privacy: switch off anonymous usage statistics.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
RAW_LAW = PROJECT / "data" / "raw" / "law"
OUT_FILE = PROJECT / "data" / "processed" / "chunks.jsonl"

# Long articles are split into parts so no single chunk is too big to search well.
MAX_CHARS = 4000

# What we know about each source file. The URL is used to build citation links.
DOCUMENTS = {
    "ai_act_2024-1689_en.html": {
        "doc_id": "ai_act_2024_1689",
        "title": "Regulation (EU) 2024/1689 (AI Act)",
        "authority": "binding_law",
        "version_date": "2024-07-12",
        "url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    },
    "ai_omnibus_2026-1744_en.html": {
        "doc_id": "ai_omnibus_2026_1744",
        "title": "Regulation (EU) 2026/1744 (Digital Omnibus on AI)",
        "authority": "binding_law",
        "version_date": "2026-07-24",
        "url": "https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng",
    },
}


def clean_text(element):
    """Get readable plain text from a piece of HTML."""
    copy = BeautifulSoup(str(element), "html.parser")
    # Footnote markers like "(1)" in superscript add noise to the text.
    for note in copy.select("span.oj-note-tag, span.oj-super"):
        note.decompose()
    text = copy.get_text(" ")
    text = text.replace("\xa0", " ")          # non-breaking spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_if_long(text, max_chars=MAX_CHARS):
    """Split very long text into parts, breaking at sentence ends where possible."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current.strip())
    return parts


def extract_units(soup):
    """Walk the document in order, yielding (kind, element_id, heading, element).

    While walking we remember the most recent chapter and section headings so
    each article can record where in the law it sits.
    """
    chapter = section = ""
    awaiting = None   # which heading the next title line belongs to

    for el in soup.find_all(["p", "div"]):
        classes = el.get("class") or []
        el_id = el.get("id", "")

        # EUR-Lex uses the same tag for "CHAPTER I" and "SECTION 1", so we look
        # at the words themselves to tell them apart.
        if "oj-ti-section-1" in classes or "oj-ti-grseq-1" in classes:
            heading = clean_text(el)
            if heading.upper().startswith("CHAPTER"):
                chapter, section, awaiting = heading, "", "chapter"
            elif heading.upper().startswith("SECTION"):
                section, awaiting = heading, "section"
            continue
        if "oj-ti-section-2" in classes and awaiting:
            # The line after a heading is its title, e.g. "GENERAL PROVISIONS".
            if awaiting == "chapter":
                chapter = f"{chapter} - {clean_text(el)}"
            else:
                section = f"{section} - {clean_text(el)}"
            awaiting = None
            continue

        if "eli-subdivision" in classes and el_id.startswith("art_"):
            heading = el.find("p", class_="oj-ti-art")
            subtitle = el.find("p", class_="oj-sti-art")
            label = clean_text(heading) if heading else el_id
            if subtitle:
                label = f"{label} - {clean_text(subtitle)}"
            yield "article", el_id, label, el, chapter, section

        elif "eli-subdivision" in classes and el_id.startswith("rct_"):
            yield "recital", el_id, f"Recital {el_id.split('_')[1]}", el, "", ""

        elif "eli-container" in classes and el_id.startswith("anx_"):
            titles = el.find_all("p", class_="oj-doc-ti", limit=2)
            label = " - ".join(clean_text(t) for t in titles) if titles else el_id
            yield "annex", el_id, label, el, "", ""


def chunks_from_file(path):
    meta = DOCUMENTS.get(path.name)
    if meta is None:
        print(f"  ! No entry in DOCUMENTS for {path.name}, skipping.")
        return []

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    chunks = []

    for kind, el_id, label, element, chapter, section in extract_units(soup):
        body = clean_text(element)
        # The heading is already in `label`; keep it at the top of the text too,
        # because search works better when the chunk names what it is about.
        parts = split_if_long(body)
        for i, part in enumerate(parts, start=1):
            suffix = f"_p{i}" if len(parts) > 1 else ""
            chunks.append({
                "chunk_id": f"{meta['doc_id']}:{el_id}{suffix}",
                "text": part,
                "doc_id": meta["doc_id"],
                "doc_title": meta["title"],
                "authority": meta["authority"],
                "version_date": meta["version_date"],
                "kind": kind,
                "label": label,
                "chapter": chapter,
                "section": section,
                "part": i,
                "of_parts": len(parts),
                "source_url": f"{meta['url']}#{el_id}",
            })
    return chunks


def main():
    if not RAW_LAW.exists():
        sys.exit(f"Could not find {RAW_LAW}. Run this from the project folder.")

    all_chunks = []
    for path in sorted(RAW_LAW.glob("*.html")):
        print(f"Reading {path.name} ...")
        file_chunks = chunks_from_file(path)
        print(f"  {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    if not all_chunks:
        sys.exit("No chunks produced. Are your .html files in data/raw/law?")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(len(c["text"]) for c in all_chunks)
    kinds = {}
    for c in all_chunks:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1

    print("\n" + "=" * 60)
    print(f"Wrote {len(all_chunks)} chunks to {OUT_FILE.relative_to(PROJECT)}")
    print("By kind: " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
    print(f"Chunk size in characters - smallest {sizes[0]}, "
          f"middle {sizes[len(sizes) // 2]}, largest {sizes[-1]}")
    print("=" * 60)
    print("\nFirst chunk as a sample:\n")
    sample = all_chunks[0]
    print(f"  id:    {sample['chunk_id']}")
    print(f"  label: {sample['label']}")
    print(f"  text:  {sample['text'][:300]}...")


if __name__ == "__main__":
    main()
