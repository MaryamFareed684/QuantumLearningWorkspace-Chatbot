# Team Mu — StudyMind Chatbot (RAG API)

Team Mu owns the chatbot under `chatbot/`. The HTTP service lives in `rag-engine/` on port **8000** (separate from `web/backend/`).

Ports per Contract v1: **8000 = Mu (rag-engine chatbot)**, 5000 = Pluto (web), 8001 = Lambda ingestion, 8002 = Lambda quiz.

## Quick start

```bash
cd chatbot
python -m venv rag_venv
rag_venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Copy env and add GROQ_API_KEY
copy rag-engine\.env.example ..\..\chatbot\.env

cd rag-engine
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Contract: [`docs/api-contracts.md`](../docs/api-contracts.md)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness, index stats, cache/rate-limit backend |
| POST | `/ask` | Sync RAG answer (cache, timing, rate limits) |
| POST | `/ask/stream` | NDJSON token stream |

> **Note on ingestion:** This repo handles retrieval and answering only. Document ingestion (chunking → embedding → ChromaDB) is handled by Team Lambda's separate `ai-ml` service on a different port. Mu does not build, deploy, or maintain an ingestion path.

## Stream test client

```bash
cd rag-engine
python scripts/stream_client.py "Where does the Calvin cycle occur?"
```

## Tests

```bash
cd chatbot
pip install -r requirements.txt
pytest rag-engine/tests -q
```

API tests use mocked RAG (no `GROQ_API_KEY` required).

## Live eval (needs Groq)

```bash
cd rag-engine
python eval/eval_suite.py --threshold 7
```

## Docker

```bash
cd chatbot
docker build -t studymind-chatbot .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key studymind-chatbot
```

Optional Redis for shared cache + rate limits:

```bash
docker run -p 8000:8000 -e GROQ_API_KEY=... -e REDIS_URL=redis://host:6379/0 studymind-chatbot
```

## Environment variables

See [`rag-engine/.env.example`](rag-engine/.env.example).

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | LLM provider (required for real answers) |
| `REDIS_URL` | Optional Redis for cache + rate limits |
| `ENABLE_CACHE` | Answer cache on/off |
| `RATE_LIMIT_MAX` | Requests per window per user |
| `ENABLE_MULTI_HOP` | Agentic retrieval hops |
| `SEED_FIXTURES` | Load local demo corpus into collection (demo/eval only, not production) |

## Project layout

```
chatbot/
├── README.md                 # this file
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── memory/                   # CLI conversation demos
└── rag-engine/
    ├── main.py               # FastAPI app
    ├── rag_service.py        # RAG pipeline
    ├── cache.py              # Answer cache (memory or Redis)
    ├── rate_limiter.py
    ├── timing_logger.py
    ├── data/                 # Demo corpus (.txt fixtures)
    ├── eval/                 # Live regression suite
    ├── scripts/              # stream_client.py
    └── tests/
```

## Upgrade roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 8 | Docker, CI, API tests, Redis backends, health readiness | **Done** |
| 9 | Persistent index (ChromaDB); `/ingest` removed — ingestion is Lambda's `ai-ml` service | **Done** |
| 10 | JWT auth (`get_current_user_email`), user-scoped retrieval, identity-from-JWT-only (P0-2) | **Done** |
| 11 | Rerank via LLM prompt (not cross-encoder); multi-hop agentic retrieval | **Basic version done** — cross-encoder rerank still possible future upgrade |
| 12 | Per-request timing (`TimingRecord` → response headers + `key=value` logs) | **Basic version done** — JSON structured logging, metrics dashboard, and runbook still upcoming |
| — | P0-5: Cache invalidation on document purge | **Upcoming** |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `503 RAG engine is not ready` | Wait for startup embedding; check `/health` `ready` |
| `503 GROQ_API_KEY` | Set key in `.env` |
| `429 Rate limit` | Wait `Retry-After` seconds; rate limit is per authenticated JWT identity |
| Slow first request | Model + index warmup on startup (expected) |
