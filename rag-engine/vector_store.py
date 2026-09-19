"""Embed chunks with SentenceTransformer and store/query them in ChromaDB."""

from __future__ import annotations
import os

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "study_chunks"
DEFAULT_TOP_K = 4
# Chroma default space is L2; lower distance = more similar.
# Tuned so on-topic photosynthesis questions pass and off-topic ones refuse.
DEFAULT_MAX_DISTANCE = 1.8


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load the local embedding model used for documents and queries."""
    return SentenceTransformer(model_name)




def create_collection(name: str = DEFAULT_COLLECTION_NAME, path: str = None):
    """Get or create a persistent Chroma collection shared across services."""
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    return client.get_or_create_collection(name=name)


def add_chunks(
    collection,
    embedding_model: SentenceTransformer,
    chunks: list[dict],
    user_id: str | None = None,
) -> None:
    """Embed each chunk and add it to the collection, optionally tagged with user_id."""
    if not chunks:
        return

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    if user_id is not None:
        metadatas = [{**chunk["metadata"], "user_id": user_id} for chunk in chunks]
    else:
        metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = embedding_model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def retrieve(
    collection,
    embedding_model: SentenceTransformer,
    question: str,
    n_results: int = DEFAULT_TOP_K,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Query the collection for the top-n chunks matching the question,
    optionally scoped to a single user_id.

    Returns a dict with:
      - documents: list[str]
      - distances: list[float] | None
      - metadatas: list[dict] | None
      - ids: list[str] | None
    """
    question_embedding = embedding_model.encode(question).tolist()
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [question_embedding],
        "n_results": n_results,
    }
    if user_id is not None:
        query_kwargs["where"] = {"user_id": user_id}

    results = collection.query(**query_kwargs)

    return {
        "documents": results["documents"][0] if results.get("documents") else [],
        "distances": results["distances"][0] if results.get("distances") else None,
        "metadatas": results["metadatas"][0] if results.get("metadatas") else None,
        "ids": results["ids"][0] if results.get("ids") else None,
    }


def is_relevant(
    distances: list[float] | None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> bool:
    """
    Return True if at least one retrieved chunk is close enough.

    Chroma returns L2 distances (lower = better). Refuse when there are no
    distances, or when the best (minimum) distance exceeds max_distance.
    """
    if not distances:
        return False
    return min(distances) <= max_distance


def user_has_documents(collection, user_id: str) -> bool:
    """
    Check whether a given user has ANY chunks stored at all,
    regardless of the question being asked.

    Used to distinguish "no documents uploaded yet" from
    "documents exist but nothing matched this question."
    """
    result = collection.get(where={"user_id": user_id}, limit=1)
    return len(result.get("ids", [])) > 0


def format_retrieved_chunks(documents: list[str]) -> str:
    """Join one or more retrieved chunks for the LLM prompt (legacy simple format)."""
    if not documents:
        return ""
    if len(documents) == 1:
        return documents[0]
    parts = []
    for i, doc in enumerate(documents, start=1):
        parts.append(f"[Chunk {i}]\n{doc}")
    return "\n\n".join(parts)


def format_untrusted_chunks(
    documents: list[str],
    ids: list[str] | None = None,
    metadatas: list[dict] | None = None,
) -> str:
    """
    Wrap chunks in clear untrusted-document delimiters for the LLM.

    Retrieved text is data to reference, never instructions to follow.
    """
    if not documents:
        return ""
    parts: list[str] = []
    for i, doc in enumerate(documents):
        chunk_id = ids[i] if ids and i < len(ids) else f"chunk_{i}"
        source = ""
        if metadatas and i < len(metadatas) and isinstance(metadatas[i], dict):
            source = str(metadatas[i].get("source") or "")
        parts.append(
            f'<<<UNTRUSTED_DOCUMENT id="{chunk_id}" source="{source}">>>\n'
            f"{doc}\n"
            f"<<<END_UNTRUSTED_DOCUMENT>>>"
        )
    return "\n\n".join(parts)


def chunk_preview(text: str, max_words: int = 20) -> str:
    """Short preview of a chunk for API source metadata."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."



