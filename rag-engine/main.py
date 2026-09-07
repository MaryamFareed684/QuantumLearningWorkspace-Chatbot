"""
Team Mu RAG Chat API — FastAPI service (separate from web/backend).

Run from chatbot/rag-engine/:
  uvicorn main:app --reload --host 127.0.0.1 --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations
from cache import answer_cache

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
from pathlib import Path

from cache import AnswerCache, answer_cache, ask_result_to_cache_entry
from rate_limiter import check_rate_limit, rate_limiter
from rag_service import (
    RagEngine,
    SourceInfo,
    ask,
    create_collection,
    create_engine,
    finalize_ask,
    generate_answer_sync,
    load_env,
    prepare_ask,
    stream_answer_tokens,
)
from schemas import AskRequest, AskResponse, HealthResponse, SourceItem, TimingInfo
from auth import get_current_user_email
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.main import app as ingestion_app

from timing_logger import TimingRecord

_engine: RagEngine | None = None
_engine_ready: bool = False


def get_engine() -> RagEngine:
    if _engine is None or not _engine_ready:
        raise HTTPException(status_code=503, detail="RAG engine is not ready yet")
    return _engine


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _skip_cache(request: Request, body: AskRequest) -> bool:
    if body.skip_cache:
        return True
    return request.headers.get("X-Skip-Cache", "").strip() in {"1", "true", "yes"}


def _user_label(request: Request, current_user_email: str | None = None) -> str:
    # Prefer JWT user email if available (from dependency injection)
    if current_user_email:
        return current_user_email.strip()
    return ""


def _history_list(body: AskRequest) -> list[dict] | None:
    if not body.history:
        return None
    return [{"role": m.role, "content": m.content} for m in body.history]


def _cache_key(body: AskRequest, user_email: str) -> str:
    return AnswerCache.make_key(
        user_email,
        body.question,
        _history_list(body),
        body.top_k,
        body.rerank,
        body.multi_hop,
        body.include_sources,
    )

    


def _sources_from_result(result: Any) -> tuple[list[SourceItem] | None, list[str]]:
    sources = None
    source_ids: list[str] = []
    if result.refused:
        return [], []
    sources = [
        SourceItem(
            id=s.id,
            distance=s.distance,
            preview=s.preview,
            source=getattr(s, "source", "") or "",
            document=getattr(s, "document", "") or "",
        )
        for s in (result.sources or [])
    ]
    source_ids = list(result.source_ids) or [s.id for s in sources]
    return sources, source_ids


def _result_to_response(
    result: Any,
    *,
    cached: bool = False,
    timing: TimingRecord | None = None,
    include_sources: bool = True,
) -> AskResponse:
    sources = None
    source_ids: list[str] = []
    if include_sources:
        if result.refused:
            sources = []
            source_ids = []
        else:
            sources, source_ids = _sources_from_result(result)

    timing_info = None
    if timing is not None:
        timing.finish()
        timing_info = TimingInfo(**timing.to_dict())

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        no_documents=result.no_documents,
        top_k=result.top_k,
        sources=sources,
        source_ids=source_ids,
        rewritten_question=result.rewritten_question,
        grounded=result.grounded,
        retrieval_rounds=result.retrieval_rounds,
        hop_queries=list(result.hop_queries),
        conflict_hint=result.conflict_hint,
        cached=cached,
        timing=timing_info,
    )


def _cache_entry_to_result(entry: Any, top_k: int) -> Any:
    from rag_service import AskResult

    sources = [
        SourceInfo(
            id=s.get("id", ""),
            distance=s.get("distance"),
            preview=s.get("preview", ""),
            source=s.get("source", ""),
            document=s.get("document", ""),
        )
        for s in entry.sources
    ]
    return AskResult(
        answer=entry.answer,
        refused=entry.refused,
        top_k=top_k,
        sources=sources,
        rewritten_question=entry.rewritten_question,
        grounded=entry.grounded,
        source_ids=list(entry.source_ids),
        retrieval_rounds=entry.retrieval_rounds,
        hop_queries=list(entry.hop_queries),
        conflict_hint=entry.conflict_hint,
    )


def _timing_headers(timing: TimingRecord | None, cached: bool) -> dict[str, str]:
    headers: dict[str, str] = {"X-Cache-Hit": "1" if cached else "0"}
    if timing is None:
        return headers
    timing.finish()
    if timing.retrieval_ms is not None:
        headers["X-Retrieval-Ms"] = str(int(timing.retrieval_ms))
    if timing.llm_ms is not None:
        headers["X-Llm-Ms"] = str(int(timing.llm_ms))
    if timing.total_ms is not None:
        headers["X-Total-Ms"] = str(int(timing.total_ms))
    return headers


def _replay_tokens(text: str) -> Iterator[str]:
    """Replay cached answer as word chunks for streaming."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if i == 0:
            yield word
        else:
            yield " " + word


def _ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _stream_ask(
    engine: RagEngine,
    body: AskRequest,
    request: Request,
    user_email: str,
) -> Iterator[str]:
    timing = TimingRecord()
    user = _user_label(request, user_email)
    skip = _skip_cache(request, body)
    history = _history_list(body)

    if not skip:
        cached = answer_cache.get(_cache_key(body, user_email))
        if cached is not None:
            timing.llm_ms = 0.0
            timing.retrieval_ms = 0.0
            timing.grounding_ms = 0.0
            timing.finish()
            meta = {
                "type": "metadata",
                "refused": cached.refused,
                "answer": cached.answer,
                "source_ids": cached.source_ids,
                "rewritten_question": cached.rewritten_question,
                "retrieval_rounds": cached.retrieval_rounds,
                "hop_queries": cached.hop_queries,
                "grounded": cached.grounded,
                "conflict_hint": cached.conflict_hint,
                "cached": True,
                "timing": timing.to_dict(),
            }
            yield _ndjson_line(meta)
            for token in _replay_tokens(cached.answer):
                yield _ndjson_line({"type": "token", "content": token})
            yield _ndjson_line(
                {
                    "type": "done",
                    "grounded": cached.grounded,
                    "cached": True,
                    "timing": timing.to_dict(),
                }
            )
            timing.log(user=user, cached=True)
            return

    try:
        if engine.auto_reconnect:
            engine.collection = create_collection(name=engine.collection.name)
        timing.start_retrieval()
        prepared = prepare_ask(
            engine,
            body.question,
            history=history,
            top_k=body.top_k,
            include_sources=body.include_sources,
            rerank=body.rerank,
            multi_hop=body.multi_hop,
            user_id=user_email,
        )
        timing.end_retrieval()
    except (ValueError, RuntimeError) as exc:
        raise exc

    source_ids = []
    if prepared.refused:
        source_ids = []
    else:
        from rag_service import _build_sources

        if body.include_sources:
            source_ids = [s.id for s in _build_sources(prepared.accumulated)]

    meta = {
        "type": "metadata",
        "refused": prepared.refused,
        "answer": prepared.refusal_answer if prepared.refused else None,
        "source_ids": source_ids,
        "rewritten_question": prepared.rewritten_question,
        "retrieval_rounds": len(prepared.hop_queries),
        "hop_queries": prepared.hop_queries,
        "grounded": None,
        "conflict_hint": False if prepared.refused else None,
        "cached": False,
        "timing": {
            **timing.to_dict(),
            "llm_ms": None,
            "total_ms": None,
        },
    }
    yield _ndjson_line(meta)

    if prepared.refused:
        timing.finish()
        yield _ndjson_line(
            {
                "type": "done",
                "grounded": None,
                "cached": False,
                "timing": timing.to_dict(),
            }
        )
        timing.log(user=user, cached=False)
        return

    client = prepared.client
    if client is None:
        from rag_service import _get_groq

        client = _get_groq(engine)

    timing.start_llm()
    answer_parts: list[str] = []
    for token in stream_answer_tokens(client, prepared):
        answer_parts.append(token)
        yield _ndjson_line({"type": "token", "content": token})
    timing.end_llm()

    answer = "".join(answer_parts)
    result = finalize_ask(engine, prepared, answer, timing=timing)

    if not skip:
        answer_cache.set(
            _cache_key(body, user_email),
            ask_result_to_cache_entry(result, include_sources=body.include_sources),
        )

    timing.finish()
    yield _ndjson_line(
        {
            "type": "done",
            "grounded": result.grounded,
            "cached": False,
            "timing": timing.to_dict(),
        }
    )
    timing.log(user=user, cached=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _engine_ready
    _engine_ready = False
    load_env()
    _engine = create_engine(collection_name="study_chunks")
    _engine_ready = True
    yield
    _engine = None
    _engine_ready = False


app = FastAPI(
    title="StudyMind Chatbot — RAG API",
    description=(
        "Team Mu chat service: streaming, caching, rate limits, multi-hop retrieval, "
        "and grounding checks. See docs/api-contracts.md."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

app.mount("/ingestion", ingestion_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    groq_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
    if not _engine_ready or _engine is None:
        return HealthResponse(
            status="warming",
            ready=False,
            chunks_indexed=0,
            embedding_model="",
            default_top_k=4,
            max_distance=1.2,
            cache_entries=answer_cache.size(),
            cache_hits=answer_cache.hits,
            cache_backend=answer_cache.backend,
            rate_limit_backend=rate_limiter.backend,
            groq_configured=groq_ok,
        )
    return HealthResponse(
        status="ok",
        ready=True,
        chunks_indexed=_engine.chunks_indexed,
        embedding_model=_engine.embedding_model_name,
        default_top_k=_engine.default_top_k,
        max_distance=_engine.max_distance,
        cache_entries=answer_cache.size(),
        cache_hits=answer_cache.hits,
        cache_backend=answer_cache.backend,
        rate_limit_backend=rate_limiter.backend,
        groq_configured=groq_ok,
    )


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(
    body: AskRequest,
    request: Request,
    response: Response,
    _: None = Depends(check_rate_limit),
    current_user_email: str = Depends(get_current_user_email),
) -> AskResponse:
    engine = get_engine()
    timing = TimingRecord()
    user = _user_label(request, current_user_email)
    skip = _skip_cache(request, body)
    history = _history_list(body)

    if not skip:
        cached = answer_cache.get(_cache_key(body, current_user_email))
        if cached is not None:
            timing.llm_ms = 0.0
            timing.retrieval_ms = 0.0
            timing.grounding_ms = 0.0
            result = _cache_entry_to_result(cached, body.top_k or engine.default_top_k)
            for k, v in _timing_headers(timing, cached=True).items():
                response.headers[k] = v
            timing.log(user=user, cached=True)
            return _result_to_response(
                result,
                cached=True,
                timing=timing,
                include_sources=body.include_sources,
            )

    try:
        if engine.auto_reconnect:
            engine.collection = create_collection(name=engine.collection.name)
        timing.start_retrieval()
        prepared = prepare_ask(
            engine,
            body.question,
            history=history,
            top_k=body.top_k,
            include_sources=body.include_sources,
            rerank=body.rerank,
            multi_hop=body.multi_hop,
            user_id=current_user_email,
        )
        timing.end_retrieval()

        if prepared.refused:
            from rag_service import _refusal_result

            result = _refusal_result(prepared)
            timing.finish()
            for k, v in _timing_headers(timing, cached=False).items():
                response.headers[k] = v
            timing.log(user=user, cached=False)
            return _result_to_response(
                result,
                cached=False,
                timing=timing,
                include_sources=body.include_sources,
            )

        client = prepared.client
        if client is None:
            from rag_service import _get_groq

            client = _get_groq(engine)

        timing.start_llm()
        answer = generate_answer_sync(client, prepared, strict=False)
        timing.end_llm()

        result = finalize_ask(engine, prepared, answer, timing=timing)

        if not skip:
            answer_cache.set(
                _cache_key(body, current_user_email),
                ask_result_to_cache_entry(result, include_sources=body.include_sources),
            )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for k, v in _timing_headers(timing, cached=False).items():
        response.headers[k] = v
    timing.log(user=user, cached=False)
    return _result_to_response(
        result,
        cached=False,
        timing=timing,
        include_sources=body.include_sources,
    )
@app.post("/ask/stream")
def ask_stream_endpoint(
    body: AskRequest,
    request: Request,
    _: None = Depends(check_rate_limit),
    current_user_email: str = Depends(get_current_user_email),
):
    engine = get_engine()

    def event_generator() -> Iterator[str]:
        try:
            yield from _stream_ask(engine, body, request, current_user_email)
        except ValueError as exc:
            yield _ndjson_line({"type": "error", "detail": str(exc)})
        except RuntimeError as exc:
            yield _ndjson_line({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


@app.delete("/internal/cache/document/{document_id}")
def invalidate_document_cache(
    document_id: str,
    current_user_email: str = Depends(get_current_user_email),
) -> dict:
    """
    [P0-5] Invalidate any cached answers whose sources came from this
    document_id. Called after Lambda purges the document from the
    shared ChromaDB store, so Mu never serves a cached answer built
    from content that no longer exists.

    Note: like Lambda's own purge endpoint, this does not verify the
    caller owns document_id — any authenticated user can invalidate
    any document's cache entries. Cache-only side effect (no data
    loss beyond needing to regenerate an answer), so low risk, but
    worth the same ownership-check note as Lambda's purge endpoint.
    """
    removed = answer_cache.invalidate_document(document_id)
    return {"success": True, "removed": removed}







