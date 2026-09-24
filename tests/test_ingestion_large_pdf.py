"""Large PDFs: chunks are upserted in batches, and PDF ingestion does not block the service."""
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import embedding.chroma_store as chroma_store  # noqa: E402


class _Vectors(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def encode(self, texts, batch_size=32):
        return _Vectors([[0.0, 0.0] for _ in texts])


class _FakeCollection:
    def __init__(self):
        self.batches = []
        self.rows = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        self.batches.append(len(ids))
        for i, chunk_id in enumerate(ids):
            self.rows[chunk_id] = metadatas[i]


def test_large_document_is_upserted_in_batches(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(chroma_store, "get_collection", lambda *a, **k: collection)
    monkeypatch.setattr(chroma_store, "get_embedding_model", lambda: _FakeModel())
    chunks = [{"chunk_index": i, "text": f"chunk {i}"} for i in range(250)]
    stored = chroma_store.store_chunks(chunks, user_id="a@x.com", document_id="doc-1", title="Big")
    assert stored == 250
    assert collection.batches == [100, 100, 50]
    assert len(collection.rows) == 250
    assert all(meta["document_id"] == "doc-1" for meta in collection.rows.values())


def test_pdf_ingestion_endpoint_runs_in_a_worker_thread():
    try:
        from ingestion.main import ingest_pdf_endpoint
    except Exception as exc:  # ingestion extras (e.g. PyMuPDF, python-multipart) not installed locally
        pytest.skip(f"ingestion dependencies not installed: {exc}")
    # A plain "def" endpoint is run by FastAPI in a thread pool, so a long PDF
    # does not block every other request to the service.
    assert not inspect.iscoroutinefunction(ingest_pdf_endpoint)
