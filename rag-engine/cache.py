"""Answer cache with in-memory LRU or optional Redis backend."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


DEFAULT_TTL_SECONDS = 900
DEFAULT_MAX_ENTRIES = 200
CACHE_KEY_PREFIX = "studymind:ask:"


@dataclass
class CacheEntry:
    """Serializable cached AskResult fields."""

    answer: str
    refused: bool
    top_k: int
    rewritten_question: str = ""
    grounded: bool | None = None
    source_ids: list[str] = field(default_factory=list)
    retrieval_rounds: int = 0
    hop_queries: list[str] = field(default_factory=list)
    conflict_hint: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    include_sources: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> CacheEntry:
        data = json.loads(raw)
        return cls(**data)


class AnswerCache:
    """LRU cache with optional Redis when REDIS_URL is set."""

    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: int | None = None,
        enabled: bool | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.max_entries = max_entries or _env_int("CACHE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)
        self.ttl_seconds = ttl_seconds or _env_int("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS)
        self.enabled = (
            enabled if enabled is not None else _env_bool("ENABLE_CACHE", True)
        )
        self._store: OrderedDict[str, tuple[float, CacheEntry]] = OrderedDict()
        self.hits = 0
        self._redis = None
        self.backend = "memory"
        url = redis_url if redis_url is not None else os.environ.get("REDIS_URL", "").strip()
        if url:
            try:
                import redis

                self._redis = redis.Redis.from_url(url, decode_responses=True)
                self._redis.ping()
                self.backend = "redis"
            except Exception:
                self._redis = None
                self.backend = "memory"

    @staticmethod
    def make_key(
        user_id: str,
        question: str,
        history: list[dict] | None,
        top_k: int | None,
        rerank: bool | None,
        multi_hop: bool | None,
        include_sources: bool | None = True,
        document_id: str | None = None,
        document_name: str | None = None,
    ) -> str:
        payload = {
            "user_id": (user_id or "").strip(),
            "question": (question or "").strip(),
            "history": history or [],
            "top_k": top_k,
            "rerank": rerank,
            "multi_hop": multi_hop,
            "include_sources": include_sources,
        }
        # Only added for document-scoped questions, so keys for unscoped
        # questions stay exactly as before and existing entries remain valid.
        if document_id:
            payload["document_id"] = document_id
        elif document_name:
            payload["document_name"] = document_name
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        if not self.enabled:
            return None
        if self._redis is not None:
            raw = self._redis.get(f"{CACHE_KEY_PREFIX}{key}")
            if raw is None:
                return None
            self.hits += 1
            return CacheEntry.from_json(raw)
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, entry = item
        if time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        if entry.refused:
            return
        if entry.grounded is False:
            return
        if self._redis is not None:
            self._redis.setex(
                f"{CACHE_KEY_PREFIX}{key}",
                self.ttl_seconds,
                entry.to_json(),
            )
            return
        expires_at = time.time() + self.ttl_seconds
        if key in self._store:
            del self._store[key]
        self._store[key] = (expires_at, entry)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def invalidate_document(self, document_id: str) -> int:
        """
        [P0-5] Remove all cached answers whose sources include this
        document_id. Cache entries store source_ids like "pdf_chunk_1",
        which are prefixed with the document_id (see chroma_store.py's
        store_chunks: f"{document_id}_chunk{index}") -- so a purged
        document's cached answers can be found and removed precisely,
        without wiping the whole cache.

        Returns the number of entries removed.
        """
        if self._redis is not None:
            removed = 0
            for key in list(self._redis.scan_iter(f"{CACHE_KEY_PREFIX}*")):
                raw = self._redis.get(key)
                if raw is None:
                    continue
                entry = CacheEntry.from_json(raw)
                if any(sid.startswith(f"{document_id}_") for sid in entry.source_ids):
                    self._redis.delete(key)
                    removed += 1
            return removed

        keys_to_remove = [
            key for key, (_, entry) in self._store.items()
            if any(sid.startswith(f"{document_id}_") for sid in entry.source_ids)
        ]
        for key in keys_to_remove:
            del self._store[key]
        return len(keys_to_remove)

    def size(self) -> int:
        if self._redis is not None:
            try:
                return len(list(self._redis.scan_iter(f"{CACHE_KEY_PREFIX}*")))
            except Exception:
                return 0
        self._evict_expired()
        return len(self._store)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]


answer_cache = AnswerCache()


DEFAULT_SUMMARY_TTL_SECONDS = 3600
DEFAULT_SUMMARY_MAX_ENTRIES = 200
SUMMARY_CACHE_KEY_PREFIX = "studymind:summary:"


class SummaryCache:
    """In-memory LRU + TTL cache for conversation summaries (Phase 11 Part C).

    Keyed by (user_id, history) so repeat /conversations/summarize calls do
    not re-hit the LLM. Deliberately memory-only (like feedback_store): the
    answer cache already has Redis plumbing, and summaries are per-process
    ephemeral data. Survives no restarts.
    """

    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.max_entries = max_entries or _env_int(
            "SUMMARY_CACHE_MAX_ENTRIES", DEFAULT_SUMMARY_MAX_ENTRIES
        )
        self.ttl_seconds = ttl_seconds or _env_int(
            "SUMMARY_CACHE_TTL_SECONDS", DEFAULT_SUMMARY_TTL_SECONDS
        )
        self.enabled = enabled if enabled is not None else _env_bool(
            "ENABLE_SUMMARY_CACHE", True
        )
        self._store: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.hits = 0

    @staticmethod
    def make_key(user_id: str, history: list[dict] | None) -> str:
        payload = {
            "user_id": (user_id or "").strip(),
            "history": history or [],
        }
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, summary = item
        if time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return summary

    def set(self, key: str, summary: str) -> None:
        if not self.enabled:
            return
        if not summary:
            return
        expires_at = time.time() + self.ttl_seconds
        if key in self._store:
            del self._store[key]
        self._store[key] = (expires_at, summary)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def size(self) -> int:
        if not self.enabled:
            return 0
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


summary_cache = SummaryCache()


def ask_result_to_cache_entry(result: Any, include_sources: bool = True) -> CacheEntry:
    """Convert AskResult to CacheEntry."""
    sources = []
    for s in getattr(result, "sources", []) or []:
        if hasattr(s, "__dataclass_fields__"):
            sources.append(asdict(s))
        elif isinstance(s, dict):
            sources.append(s)
        else:
            sources.append(
                {
                    "id": getattr(s, "id", ""),
                    "preview": getattr(s, "preview", ""),
                    "source": getattr(s, "source", ""),
                    "document": getattr(s, "document", ""),
                    "page": getattr(s, "page", None),
                }
            )
    return CacheEntry(
        answer=result.answer,
        refused=result.refused,
        top_k=result.top_k,
        rewritten_question=result.rewritten_question,
        grounded=result.grounded,
        source_ids=list(result.source_ids),
        retrieval_rounds=result.retrieval_rounds,
        hop_queries=list(result.hop_queries),
        conflict_hint=result.conflict_hint,
        sources=sources,
        include_sources=include_sources,
    )