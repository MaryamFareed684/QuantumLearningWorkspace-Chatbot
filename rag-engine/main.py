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
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
from pathlib import Path

from cache import AnswerCache, answer_cache, ask_result_to_cache_entry, summary_cache
from rate_limiter import check_rate_limit, rate_limiter
from rag_service import (
    RagEngine,
    SourceInfo,
    _clarification_result,
    _get_groq,
    ask,
    create_collection,
    create_engine,
    finalize_ask,
    generate_answer_sync,
    load_env,
    prepare_ask,
    stream_answer_tokens,
    summarize_conversation,
)
from schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    SourceItem,
    SummarizeRequest,
    SummarizeResponse,
    TimingInfo,
)
from auth import get_current_user_email
from feedback_store import feedback_store, record_from_response
from request_logger import log_request
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


def _count_turns(history: list[dict] | None) -> int:
    """Number of user-role turns in a history list (assistant-only ignored)."""
    return sum(1 for m in (history or []) if m.get("role") == "user")


def _cache_key(body: AskRequest, user_email: str) -> str:
    # TODO(Phase 11 Part C — raise with Maryam/TL before changing): this key is
    # hashed from the client-supplied raw history (body.history), NOT from the
    # server-side compacted history used in the prompt. So compaction in
    # prepare_ask() does not change the key or break existing cache entries.
    # However it DOES mean two conversations that differ only in turns older
    # than HISTORY_TURN_CAP produce different keys yet virtually identical
    # effective context after compaction — i.e. the key can over-differentiate
    # and lower the hit rate as conversations grow. A more compact key (recent
    # turns + summary fingerprint) would change invalidation/collision behavior
    # elsewhere, so it needs sign-off before touching.
    return AnswerCache.make_key(
        user_email,
        body.question,
        _history_list(body),
        body.top_k,
        body.rerank,
        body.multi_hop,
        body.include_sources,
        document_id=body.document_id,
        document_name=body.filename,
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
            page=getattr(s, "page", None),
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
    response_id: str = "",
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
        response_id=response_id,
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


def _snapshot_response(
    *,
    response_id: str,
    user: str,
    question: str,
    answer: str,
    source_ids: list[str],
) -> None:
    """Store the /ask response so a later /feedback call can attach a rating."""
    feedback_store.put(
        record_from_response(
            response_id=response_id,
            user_id=user,
            question=question,
            answer=answer,
            source_ids=source_ids,
        )
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
            page=s.get("page"),
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
    response_id = uuid4().hex

    if not skip:
        cached = answer_cache.get(_cache_key(body, user_email))
        if cached is not None:
            timing.llm_ms = 0.0
            timing.retrieval_ms = 0.0
            timing.grounding_ms = 0.0
            timing.finish()
            meta = {
                "type": "metadata",
                "response_id": response_id,
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
                    "response_id": response_id,
                    "grounded": cached.grounded,
                    "cached": True,
                    "timing": timing.to_dict(),
                }
            )
            _snapshot_response(
                response_id=response_id,
                user=user,
                question=body.question,
                answer=cached.answer,
                source_ids=cached.source_ids,
            )
            log_request(
                endpoint="/ask/stream",
                user_id=user,
                question=body.question,
                timing=timing,
                grounded=cached.grounded,
                cached=True,
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
            document_id=body.document_id,
            document_name=body.filename,
        )
        timing.end_retrieval()
    except (ValueError, RuntimeError) as exc:
        log_request(
            endpoint="/ask/stream",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
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
        "response_id": response_id,
        "refused": prepared.refused,
        "answer": (
            prepared.clarification_message
            if prepared.clarification_required
            else (prepared.refusal_answer if prepared.refused else None)
        ),
        "source_ids": source_ids,
        "rewritten_question": prepared.rewritten_question,
        "retrieval_rounds": len(prepared.hop_queries),
        "hop_queries": prepared.hop_queries,
        "grounded": None,
        "conflict_hint": False
        if (prepared.refused or prepared.clarification_required)
        else None,
        "is_clarification": prepared.clarification_required,
        "cached": False,
        "timing": {
            **timing.to_dict(),
            "llm_ms": None,
            "total_ms": None,
        },
    }
    yield _ndjson_line(meta)

    if prepared.clarification_required:
        timing.finish()
        yield _ndjson_line(
            {
                "type": "done",
                "response_id": response_id,
                "grounded": None,
                "cached": False,
                "timing": timing.to_dict(),
            }
        )
        _snapshot_response(
            response_id=response_id,
            user=user,
            question=body.question,
            answer=prepared.clarification_message,
            source_ids=[],
        )
        log_request(
            endpoint="/ask/stream",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
        timing.log(user=user, cached=False)
        return

    if prepared.refused:
        timing.finish()
        yield _ndjson_line(
            {
                "type": "done",
                "response_id": response_id,
                "grounded": None,
                "cached": False,
                "timing": timing.to_dict(),
            }
        )
        _snapshot_response(
            response_id=response_id,
            user=user,
            question=body.question,
            answer=prepared.refusal_answer,
            source_ids=[],
        )
        log_request(
            endpoint="/ask/stream",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
        timing.log(user=user, cached=False)
        return

    client = prepared.client
    if client is None:
        from rag_service import _get_groq

        client = _get_groq(engine)

    timing.start_llm()
    answer_parts: list[str] = []
    try:
        for token in stream_answer_tokens(client, prepared):
            answer_parts.append(token)
            yield _ndjson_line({"type": "token", "content": token})
    except (ValueError, RuntimeError) as exc:
        log_request(
            endpoint="/ask/stream",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
        raise exc from exc
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
            "response_id": response_id,
            "grounded": result.grounded,
            "cached": False,
            "timing": timing.to_dict(),
        }
    )
    _snapshot_response(
        response_id=response_id,
        user=user,
        question=body.question,
        answer=result.answer,
        source_ids=result.source_ids,
    )
    log_request(
        endpoint="/ask/stream",
        user_id=user,
        question=body.question,
        timing=timing,
        grounded=result.grounded,
        cached=False,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "Chatbot is running!"}


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
    response_id = uuid4().hex

    if not skip:
        cached = answer_cache.get(_cache_key(body, current_user_email))
        if cached is not None:
            timing.llm_ms = 0.0
            timing.retrieval_ms = 0.0
            timing.grounding_ms = 0.0
            result = _cache_entry_to_result(cached, body.top_k or engine.default_top_k)
            for k, v in _timing_headers(timing, cached=True).items():
                response.headers[k] = v
            log_request(
                endpoint="/ask",
                user_id=user,
                question=body.question,
                timing=timing,
                grounded=result.grounded,
                cached=True,
            )
            timing.log(user=user, cached=True)
            _snapshot_response(
                response_id=response_id,
                user=user,
                question=body.question,
                answer=result.answer,
                source_ids=result.source_ids,
            )
            return _result_to_response(
                result,
                cached=True,
                timing=timing,
                include_sources=body.include_sources,
                response_id=response_id,
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
            document_id=body.document_id,
            document_name=body.filename,
        )
        timing.end_retrieval()

        if prepared.clarification_required:
            result = _clarification_result(prepared)
            timing.finish()
            for k, v in _timing_headers(timing, cached=False).items():
                response.headers[k] = v
            log_request(
                endpoint="/ask",
                user_id=user,
                question=body.question,
                timing=timing,
                grounded=result.grounded,
                cached=False,
            )
            timing.log(user=user, cached=False)
            _snapshot_response(
                response_id=response_id,
                user=user,
                question=body.question,
                answer=result.answer,
                source_ids=[],
            )
            return _result_to_response(
                result,
                cached=False,
                timing=timing,
                include_sources=body.include_sources,
                response_id=response_id,
            )

        if prepared.refused:
            from rag_service import _refusal_result

            result = _refusal_result(prepared)
            timing.finish()
            for k, v in _timing_headers(timing, cached=False).items():
                response.headers[k] = v
            log_request(
                endpoint="/ask",
                user_id=user,
                question=body.question,
                timing=timing,
                grounded=result.grounded,
                cached=False,
            )
            timing.log(user=user, cached=False)
            _snapshot_response(
                response_id=response_id,
                user=user,
                question=body.question,
                answer=result.answer,
                source_ids=result.source_ids,
            )
            return _result_to_response(
                result,
                cached=False,
                timing=timing,
                include_sources=body.include_sources,
                response_id=response_id,
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
        log_request(
            endpoint="/ask",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log_request(
            endpoint="/ask",
            user_id=user,
            question=body.question,
            timing=timing,
            grounded=None,
            cached=False,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for k, v in _timing_headers(timing, cached=False).items():
        response.headers[k] = v
    log_request(
        endpoint="/ask",
        user_id=user,
        question=body.question,
        timing=timing,
        grounded=result.grounded,
        cached=False,
    )
    timing.log(user=user, cached=False)
    _snapshot_response(
        response_id=response_id,
        user=user,
        question=body.question,
        answer=result.answer,
        source_ids=result.source_ids,
    )
    return _result_to_response(
        result,
        cached=False,
        timing=timing,
        include_sources=body.include_sources,
        response_id=response_id,
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


@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(
    body: FeedbackRequest,
    _: None = Depends(check_rate_limit),
    current_user_email: str = Depends(get_current_user_email),
) -> FeedbackResponse:
    """Attach a thumbs up/down to a previously returned /ask response_id."""
    record = feedback_store.submit(body.response_id, body.rating)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No answer found for response_id {body.response_id!r}",
        )
    return FeedbackResponse(
        response_id=record.response_id,
        rating=body.rating,
        received=True,
    )


@app.post("/conversations/summarize", response_model=SummarizeResponse)
def summarize_conversation_endpoint(
    body: SummarizeRequest,
    _: None = Depends(check_rate_limit),
    current_user_email: str = Depends(get_current_user_email),
) -> SummarizeResponse:
    """Summarize a client-supplied conversation for study review.

    Read-only operation: the history in the body is summarized by the LLM
    using a dedicated summarization prompt (no retrieval). Posting the same
    history again serves the cached summary without another LLM call.
    """
    engine = get_engine()
    history = [{"role": m.role, "content": m.content} for m in body.history]
    cache_key = summary_cache.make_key(current_user_email, history)
    cached_summary = summary_cache.get(cache_key)
    if cached_summary is not None:
        return SummarizeResponse(
            summary=cached_summary,
            turn_count=_count_turns(history),
            cached=True,
        )
    client = _get_groq(engine)
    summary = summarize_conversation(client, history)
    if not summary or not summary.strip():
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an empty summary",
        )
    summary_cache.set(cache_key, summary)
    return SummarizeResponse(
        summary=summary,
        turn_count=_count_turns(history),
        cached=False,
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







