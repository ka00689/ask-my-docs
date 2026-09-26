"""Step 2 of ingestion: turn the chunks into embeddings and store them in Chroma.

Run it from the project folder with the virtual environment active:
    python src/build_index.py

Reads data/processed/chunks.jsonl and writes a searchable database into
data/index/chroma. Safe to run again: it rebuilds the collection from scratch.
"""
import json
import os
import sys
from pathlib import Path

# Privacy: switch off anonymous usage statistics.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT / "data" / "index" / "chroma"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ai_act"
BATCH_SIZE = 200

# Fields worth keeping alongside each chunk, used later for citations and filters.
METADATA_FIELDS = ["doc_id", "doc_title", "authority", "version_date",
                   "kind", "label", "chapter", "section", "part", "of_parts",
                   "source_url"]


def load_chunks():
    if not CHUNKS_FILE.exists():
        sys.exit(f"{CHUNKS_FILE} not found. Run: python src/extract_chunks.py")
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    if not chunks:
        sys.exit("chunks.jsonl is empty.")
    return chunks


def text_to_embed(chunk):
    """What we actually turn into numbers.

    The label is included so that a chunk about, say, Article 5 carries the
    words "Article 5 Prohibited AI practices" even if the body never repeats them.
    """
    return f"{chunk['label']}\n{chunk['text']}"


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading the embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Turning chunks into embeddings. This takes a couple of minutes...")
    embeddings = model.encode(
        [text_to_embed(c) for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    # Start clean so re-running never leaves stale or duplicated chunks behind.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity suits these embeddings
    )

    print("Storing them in Chroma...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=[embeddings[start + i].tolist() for i in range(len(batch))],
            metadatas=[{f: c[f] for f in METADATA_FIELDS} for c in batch],
        )
        print(f"  stored {min(start + BATCH_SIZE, len(chunks))} of {len(chunks)}")

    print("\n" + "=" * 60)
    print(f"Index built: {collection.count()} chunks in {INDEX_DIR.relative_to(PROJECT)}")
    print("=" * 60)
    print("\nNext: try a search with")
    print('   python src/search.py "When do high-risk obligations apply?"')


if __name__ == "__main__":
    main()
