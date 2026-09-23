"""Document scoping for /ask: Chroma filters, cache keys and source alignment."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache import AnswerCache  # noqa: E402
from rag_service import merge_results  # noqa: E402
from vector_store import build_scope_where  # noqa: E402


def test_scope_user_only():
    assert build_scope_where("a@x.com") == {"user_id": "a@x.com"}


def test_scope_user_and_document():
    assert build_scope_where("a@x.com", "doc-1") == {
        "$and": [{"user_id": "a@x.com"}, {"document_id": "doc-1"}]
    }


def test_scope_filename_fallback():
    assert build_scope_where("a@x.com", None, "rag.pdf") == {
        "$and": [{"user_id": "a@x.com"}, {"document": "rag.pdf"}]
    }


def test_scope_document_id_wins_over_filename():
    where = build_scope_where("a@x.com", "doc-1", "rag.pdf")
    assert {"document_id": "doc-1"} in where["$and"]
    assert {"document": "rag.pdf"} not in where["$and"]


def test_scope_nothing():
    assert build_scope_where() is None


def _key(**extra):
    return AnswerCache.make_key("a@x.com", "what is it?", [], 4, False, False, True, **extra)


def test_cache_key_unscoped_unchanged():
    assert _key() == _key(document_id=None, document_name=None)


def test_cache_key_differs_per_document():
    assert _key(document_id="doc-lr") != _key(document_id="doc-rag")
    assert _key(document_id="doc-lr") != _key()
    assert _key(document_name="lr.pdf") != _key(document_name="rag.pdf")


def test_merge_results_keeps_metadata_aligned():
    first = {
        "documents": ["lr text"],
        "distances": [0.1],
        "ids": ["lr_chunk0"],
        "metadatas": [{"document": "linear_regression.pdf"}],
    }
    second = {
        "documents": ["rag text a", "rag text b"],
        "distances": [0.2, 0.3],
        "ids": ["rag_chunk0", "rag_chunk1"],
        "metadatas": [{"document": "rag.pdf"}],  # one metadata entry missing
    }
    merged = merge_results(first, second)
    assert len(merged["metadatas"]) == len(merged["documents"]) == 3
    assert merged["metadatas"][0] == {"document": "linear_regression.pdf"}
    assert merged["metadatas"][1] == {"document": "rag.pdf"}
    assert merged["metadatas"][2] == {}
