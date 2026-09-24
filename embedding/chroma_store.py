"""
Shared ChromaDB store — the single canonical content store for both
the ingestion pipeline and quiz generation.

[P0-1 fix] Previously, quiz generation queried Pinecone + MongoDB
(see the old embedder.py) while ingestion wrote here. The two never
overlapped, so newly ingested content was invisible to quiz
generation. This file is now the ONLY storage path for both:
ingestion calls store_chunks() (unchanged), and quiz generation's
Embedder.search() calls query_chunks() (new) instead of touching
Pinecone/Mongo at all.

Also fixes a subtler bug: this module used to load its own separate
SentenceTransformer instance with default (non-normalized) output,
while embedding/model.py's shared model normalizes its vectors. Two
different embedding configs writing into the same cosine-similarity
space would have produced quietly wrong nearest-neighbor results.
Both reads and writes now go through the one shared, normalized
model in model.py.
"""
from __future__ import annotations
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from embedding.model import get_embedding_model

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

DEFAULT_CHROMA_PATH = os.getenv(
    "CHROMA_DB_PATH",
    r"C:\Dev\QuantumLearningWorkspace\shared_chroma_data",
)
DEFAULT_COLLECTION_NAME = "study_chunks"
# A large PDF produces hundreds of chunks; upserting them in one request can
# exceed the vector store's per-request limits, so they are sent in batches.
UPSERT_BATCH_SIZE = 100


def get_collection(name: str = DEFAULT_COLLECTION_NAME, path: str = None):
    """
    Chroma Cloud when CHROMA_API_KEY is set: the same store the chat
    (rag-engine), quiz, roadmap and knowledge-graph services read from.

    This used to be a local PersistentClient, so in production every upload was
    written to the ingestion container's own disk: invisible to all other
    services and wiped on each redeploy. The local store is now only used for
    local development (no CHROMA_API_KEY) or when a path is passed explicitly.
    """
    if path is None and os.getenv("CHROMA_API_KEY"):
        client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
    else:
        local_path = path or DEFAULT_CHROMA_PATH
        print(
            f"CHROMA_WARNING: using a local ChromaDB at {local_path!r}. Other services "
            "read from Chroma Cloud; set CHROMA_API_KEY, CHROMA_TENANT and "
            "CHROMA_DATABASE in production.",
            flush=True,
        )
        client = chromadb.PersistentClient(path=local_path)
    return client.get_or_create_collection(name=name)


def store_chunks(chunks: list[dict], user_id: str, document_id: str, title: str) -> int:
    """
    Embed and store chunks in the shared ChromaDB collection.

    chunks: list of dicts from chunker.chunk_document(), each with a "text" key.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    collection = get_collection()
    model = get_embedding_model()  # shared, normalized model — same one queries use

    ids = [f"{document_id}_chunk{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "user_id": user_id,
            "document_id": document_id,
            "document": title,
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    embeddings = model.encode(documents)  # list[list[float]], already normalized

    try:
        for start in range(0, len(ids), UPSERT_BATCH_SIZE):
            end = start + UPSERT_BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        print(f"CHROMA_UPSERT_OK: stored {len(chunks)} chunks for user_id={user_id!r} document_id={document_id}", flush=True)
    except Exception as e:
        print(f"CHROMA_UPSERT_ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    return len(chunks)


def query_chunks(
    query_text: str,
    top_k: int = 5,
    user_id: str = None,
    document_id: str = None,
) -> list:
    """
    [P0-1 new] Embed a query and return the top_k nearest chunks from
    the shared collection, in the shape quiz_service.py expects:
        [{"score": float, "text": str, "title": str, "metadata": dict}, ...]

    user_id / document_id, if given, are applied as an exact-match
    metadata filter (ChromaDB's `where`). Both default to None (no
    filter) today, which preserves current unscoped behavior — this
    is deliberately plumbed through now so P0-3 (user-scoped quiz
    retrieval) can pass user_id here without another storage change.
    """
    collection = get_collection()
    model = get_embedding_model()

    where = {}
    if user_id is not None:
        where["user_id"] = user_id
    if document_id is not None:
        where["document_id"] = document_id

    query_embedding = model.encode(query_text)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where or None,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        output.append({
            "score": distances[i],
            "text": documents[i],
            "title": meta.get("document", ""),
            "metadata": meta,
        })
    return output


def delete_chunks(document_id: str) -> None:
    """
    [P0-1 new] Remove all chunks belonging to a document from the
    shared collection. Chroma's delete-by-filter is idempotent
    (deleting a non-existent id/filter is a no-op), which is also
    what P0-5's purge endpoint will need.
    """
    collection = get_collection()
    collection.delete(where={"document_id": document_id})
