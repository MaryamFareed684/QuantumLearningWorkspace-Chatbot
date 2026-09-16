"""Unit tests for _retrieve_round() method dispatch in the RAG pipeline.

Covers which retrieval backend is used (semantic vs hybrid, driven by the
RETRIEVAL_METHOD env var / explicit kwarg) and when the LLM re-ranking step
kicks in (candidate count > top_k) versus the plain top-k select.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import RERANK_CANDIDATE_COUNT, RagEngine, _retrieve_round  # noqa: E402


def _engine(chunks_indexed: int = 10) -> RagEngine:
    return RagEngine(
        collection=MagicMock(),
        embedding_model=MagicMock(),
        chunks_indexed=chunks_indexed,
    )


def _results(n: int) -> dict:
    return {
        "documents": [f"doc {i}" for i in range(n)],
        "distances": [float(i) for i in range(n)],
        "ids": [f"c{i}" for i in range(n)],
        "metadatas": [{"i": i} for i in range(n)],
    }


def test_retrieve_round_dispatches_semantic_by_default(monkeypatch):
    engine = _engine()
    captured = {}

    def fake_retrieve(collection, embedding_model, question, n_results=None, user_id=None):
        captured["args"] = (collection, embedding_model, question, n_results, user_id)
        return _results(2)

    def fail_hybrid(*args, **kwargs):
        raise AssertionError("hybrid_retrieve must not be called in semantic mode")

    monkeypatch.setattr("rag_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag_service.hybrid_retrieve", fail_hybrid)
    monkeypatch.delenv("RETRIEVAL_METHOD", raising=False)

    out, client = _retrieve_round(engine, None, "what is ATP?", k=2, do_rerank=False)

    assert captured["args"][1] is engine.embedding_model
    assert captured["args"][2] == "what is ATP?"
    assert captured["args"][3] == 2
    assert captured["args"][4] is None
    assert out["ids"] == ["c0", "c1"]
    assert client is None


def test_retrieve_round_dispatches_hybrid_from_env(monkeypatch):
    engine = _engine()
    captured = {}

    def fail_semantic(*args, **kwargs):
        raise AssertionError("retrieve must not be called in hybrid mode")

    def fake_hybrid(engine_arg, question, n_results=None, user_id=None):
        captured["args"] = (engine_arg, question, n_results, user_id)
        return {**_results(2), "rrf_scores": [0.15, 0.12]}

    monkeypatch.setattr("rag_service.retrieve", fail_semantic)
    monkeypatch.setattr("rag_service.hybrid_retrieve", fake_hybrid)
    monkeypatch.setenv("RETRIEVAL_METHOD", "hybrid")

    out, client = _retrieve_round(engine, None, "atp", k=2, do_rerank=False)

    assert captured["args"][0] is engine
    assert captured["args"][1] == "atp"
    assert captured["args"][2] == 2
    assert captured["args"][3] is None
    assert out["ids"] == ["c0", "c1"]
    assert client is None


def test_retrieve_round_triggers_rerank_when_candidates_exceed_k(monkeypatch):
    engine = _engine(chunks_indexed=10)
    client = MagicMock()
    captured = {}

    def fake_retrieve(collection, embedding_model, question, n_results=None, user_id=None):
        captured["n"] = n_results
        return _results(10)

    def fake_rerank(cli, question, results, top_k):
        captured["rerank"] = (cli, question, len(results.get("ids") or []), top_k)
        return _results(3)

    monkeypatch.setattr("rag_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag_service.rerank_chunks", fake_rerank)
    monkeypatch.delenv("RETRIEVAL_METHOD", raising=False)

    out, out_client = _retrieve_round(engine, client, "what is ATP?", k=3, do_rerank=True)

    assert captured["n"] == RERANK_CANDIDATE_COUNT
    assert captured["rerank"][0] is client
    assert captured["rerank"][1] == "what is ATP?"
    assert captured["rerank"][2] == 10
    assert captured["rerank"][3] == 3
    assert out["ids"] == ["c0", "c1", "c2"]
    assert out_client is client


def test_retrieve_round_skips_rerank_when_candidates_within_k(monkeypatch):
    engine = _engine(chunks_indexed=10)

    def fake_retrieve(collection, embedding_model, question, n_results=None, user_id=None):
        return _results(3)

    def fail_rerank(*args, **kwargs):
        raise AssertionError("rerank_chunks must not be called when candidates <= k")

    monkeypatch.setattr("rag_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag_service.rerank_chunks", fail_rerank)
    monkeypatch.delenv("RETRIEVAL_METHOD", raising=False)

    out, client = _retrieve_round(engine, None, "q", k=3, do_rerank=True)

    assert out["ids"] == ["c0", "c1", "c2"]
    assert client is None