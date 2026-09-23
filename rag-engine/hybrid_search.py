"""Hybrid retrieval path (Phase 11 Task 1).

Fuses ChromaDB semantic search (MiniLM embeddings) with the local BM25 lexical
index using Reciprocal Rank Fusion (RRF). Downstream LLM re-ranking is unchanged,
so hybrid vs semantic can be A/B'd while holding re-ranking constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bm25 import BM25Index
from vector_store import DEFAULT_TOP_K, build_scope_where, retrieve

RRF_CONSTANT = 60.0
# How many candidates each side contributes before fusion.
DEFAULT_CANDIDATE_POOL = 20
# Cap on corpus size pulled into the lexical index (local/demo corpus is small).
LEXICAL_MAX_DOCS = 20_000


@dataclass
class LexicalIndex:
    """BM25 index plus the doc text/metadata maps needed to look up hits."""

    bm25: BM25Index
    docs: dict[str, str] = field(default_factory=dict)
    metadatas: dict[str, dict] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.bm25.size


def _collection_fingerprint(
    collection,
    user_id: str | None,
    limit: int = LEXICAL_MAX_DOCS,
    document_id: str | None = None,
    document_name: str | None = None,
) -> tuple:
    """Read all scoped rows once; returns a (fingerprint, rows) pair.

    The fingerprint is the sorted id set so the cache can cheaply detect
    indexing changes (new uploads) without re-building the whole index.
    """
    where = build_scope_where(user_id or None, document_id, document_name)
    rows = collection.get(
        where=where,
        include=["documents", "metadatas"],
        limit=limit,
    )
    ids = list(rows.get("ids") or [])
    fingerprint = (collection.name, tuple(sorted(ids)))
    return fingerprint, rows


def get_bm25_index(
    engine,
    user_id: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
) -> LexicalIndex:
    """Build (or retrieve cached) lexical index for the engine's collection."""
    collection = engine.collection
    coll_name = getattr(collection, "name", "?")
    fingerprint, rows = _collection_fingerprint(
        collection, user_id, document_id=document_id, document_name=document_name
    )
    cached = getattr(engine, "_lexical_cache", None)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    ids = list(rows.get("ids") or [])
    docs = list(rows.get("documents") or [])
    metadatas = list(rows.get("metadatas") or [])

    paired: list[tuple[str, str]] = []
    for i, cid in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        paired.append((cid, text or ""))

    bm25 = BM25Index.build(paired)
    lex = LexicalIndex(
        bm25=bm25,
        docs={cid: (docs[i] if i < len(docs) else "") for i, cid in enumerate(ids)},
        metadatas={
            cid: (metadatas[i] if i < len(metadatas) else {}) for i, cid in enumerate(ids)
        },
    )
    engine._lexical_cache = (fingerprint, lex)
    return lex


def rrf_scores(
    semantic_ids: list[str],
    lexical_ids: list[str],
    k: float = RRF_CONSTANT,
) -> dict[str, float]:
    """Reciprocal rank fusion scores (higher = better) over two ranked lists."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(semantic_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(lexical_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def rrf_fuse(
    semantic_ids: list[str],
    lexical_ids: list[str],
    k: float = RRF_CONSTANT,
) -> list[str]:
    """Return fused doc ids, best first."""
    scores = rrf_scores(semantic_ids, lexical_ids, k)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def hybrid_retrieve(
    engine,
    question: str,
    n_results: int = DEFAULT_TOP_K,
    user_id: str | None = None,
    semantic_pool: int | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    lexical_pool: int | None = None,
) -> dict[str, Any]:
    """
    Retrieve via semantic + BM25 fused with RRF.

    Returns the same dict shape as vector_store.retrieve() so downstream
    (LLM re-ranking, gating, merging) is unchanged:
      - documents / distances / metadatas / ids
        distances hold the *semantic* L2 distance when the chunk came from the
        semantic pool and None for lexical-only hits.
      - rrf_scores: fused RRF scores (for introspection / reports).
    """
    pool = semantic_pool or DEFAULT_CANDIDATE_POOL
    lex_pool = lexical_pool or DEFAULT_CANDIDATE_POOL

    semantic = retrieve(
        engine.collection,
        engine.embedding_model,
        question,
        n_results=pool,
        user_id=user_id,
        document_id=document_id,
        document_name=document_name,
    )
    sem_ids = list(semantic.get("ids") or [])

    lex = get_bm25_index(
        engine, user_id=user_id, document_id=document_id, document_name=document_name
    )
    ranked = sorted(lex.bm25.score(question).items(), key=lambda kv: kv[1], reverse=True)
    lex_ids = [cid for cid, _score in ranked[:lex_pool]]

    fused_ids = rrf_fuse(sem_ids, lex_ids)[:n_results]
    fused_scores = rrf_scores(sem_ids, lex_ids)

    sem_docs = list(semantic.get("documents") or [])
    sem_dists = semantic.get("distances")
    sem_metas = list(semantic.get("metadatas") or [])
    sem_idx = {cid: i for i, cid in enumerate(sem_ids)}

    documents: list[str] = []
    distances: list[float | None] = []
    metadatas: list[dict] = []
    rrf: list[float] = []
    for cid in fused_ids:
        rrf.append(fused_scores.get(cid, 0.0))
        i = sem_idx.get(cid)
        if i is not None:
            documents.append(sem_docs[i] if i < len(sem_docs) else "")
            distances.append(sem_dists[i] if sem_dists is not None and i < len(sem_dists) else None)
            metadatas.append(sem_metas[i] if i < len(sem_metas) else {})
        else:
            documents.append(lex.docs.get(cid, ""))
            distances.append(None)
            metadatas.append(lex.metadatas.get(cid) or {})

    return {
        "documents": documents,
        "distances": distances,
        "ids": fused_ids,
        "metadatas": metadatas,
        "rrf_scores": rrf,
    }


def is_hybrid_relevant(results: dict[str, Any], max_distance: float) -> bool:
    """
    Hybrid-aware relevance gate for the first retrieval round.

    A query is on-topic when either (a) a fused chunk came from the semantic
    pool with L2 distance <= max_distance, or (b) a fused chunk is lexical-only
    (real keyword overlap with the corpus). Off-topic queries share no content
    words with the corpus, so BM25 contributes nothing and the gate refuses.
    """
    ids = results.get("ids") or []
    if not ids:
        return False
    distances = results.get("distances") or []
    real = [d for d in distances if d is not None]
    if real and min(real) <= max_distance:
        return True
    return len(real) < len(ids)