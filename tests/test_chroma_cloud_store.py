"""Ingestion must write to Chroma Cloud (the store chat, quiz, roadmap and the
knowledge graph read from), not to the container's local disk."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import embedding.chroma_store as chroma_store  # noqa: E402


class _FakeClient:
    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs

    def get_or_create_collection(self, name):
        return {"kind": self.kind, "name": name, **self.kwargs}


def _patch_clients(monkeypatch):
    monkeypatch.setattr(chroma_store.chromadb, "CloudClient", lambda **kw: _FakeClient("cloud", **kw))
    monkeypatch.setattr(chroma_store.chromadb, "PersistentClient", lambda **kw: _FakeClient("local", **kw))


def test_uses_chroma_cloud_when_credentials_are_set(monkeypatch):
    _patch_clients(monkeypatch)
    monkeypatch.setenv("CHROMA_API_KEY", "key")
    monkeypatch.setenv("CHROMA_TENANT", "tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "studymind-prod")
    collection = chroma_store.get_collection()
    assert collection["kind"] == "cloud"
    assert collection["tenant"] == "tenant" and collection["database"] == "studymind-prod"
    assert collection["name"] == "study_chunks"


def test_falls_back_to_local_without_credentials(monkeypatch, capsys):
    _patch_clients(monkeypatch)
    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    assert chroma_store.get_collection()["kind"] == "local"
    assert "CHROMA_WARNING" in capsys.readouterr().out


def test_explicit_path_stays_local(monkeypatch):
    _patch_clients(monkeypatch)
    monkeypatch.setenv("CHROMA_API_KEY", "key")
    assert chroma_store.get_collection(path="./tmp_chroma")["kind"] == "local"
