"""add_user_document: different files never share chunk ids; chunks carry document_id."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rag_service  # noqa: E402

TEXT_A = "Linear regression fits a line to data by minimising squared residuals. " * 30
TEXT_B = "Retrieval augmented generation grounds answers in retrieved documents. " * 30


def _add(monkeypatch, path, user="a@x.com"):
    captured = {}

    def fake_add_chunks(collection, embedding_model, chunks, user_id=None):
        captured["chunks"] = chunks
        captured["user_id"] = user_id

    monkeypatch.setattr(rag_service, "add_chunks", fake_add_chunks)
    engine = SimpleNamespace(collection=None, embedding_model=None, chunks_indexed=0)
    rag_service.add_user_document(engine, path, user_id=user)
    return captured


def test_two_documents_of_the_same_type_never_share_chunk_ids(monkeypatch, tmp_path):
    lr = tmp_path / "linear_regression.txt"
    rag = tmp_path / "rag.txt"
    lr.write_text(TEXT_A)
    rag.write_text(TEXT_B)
    a, b = _add(monkeypatch, lr), _add(monkeypatch, rag)
    ids_a = {c["id"] for c in a["chunks"]}
    ids_b = {c["id"] for c in b["chunks"]}
    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)


def test_chunks_are_tagged_with_document_id_and_name(monkeypatch, tmp_path):
    lr = tmp_path / "linear_regression.txt"
    lr.write_text(TEXT_A)
    captured = _add(monkeypatch, lr)
    metas = [c["metadata"] for c in captured["chunks"]]
    doc_ids = {m["document_id"] for m in metas}
    assert len(doc_ids) == 1 and all(m["document"] == "linear_regression" for m in metas)
    doc_id = next(iter(doc_ids))
    assert all(c["id"].startswith(f"{doc_id}_") for c in captured["chunks"])
    assert captured["user_id"] == "a@x.com"


def test_re_adding_the_same_file_reuses_ids(monkeypatch, tmp_path):
    lr = tmp_path / "linear_regression.txt"
    lr.write_text(TEXT_A)
    first = [c["id"] for c in _add(monkeypatch, lr)["chunks"]]
    second = [c["id"] for c in _add(monkeypatch, lr)["chunks"]]
    assert first == second  # still an upsert, no duplicates after a restart


def test_same_file_for_two_users_gets_different_ids(monkeypatch, tmp_path):
    lr = tmp_path / "linear_regression.txt"
    lr.write_text(TEXT_A)
    ids_a = {c["id"] for c in _add(monkeypatch, lr, user="a@x.com")["chunks"]}
    ids_b = {c["id"] for c in _add(monkeypatch, lr, user="b@x.com")["chunks"]}
    assert ids_a.isdisjoint(ids_b)
