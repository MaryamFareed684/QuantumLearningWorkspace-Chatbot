
## Phase 11 at a Glance

Phase 11 wrapped up the retrieval-quality and conversation-quality work on top of the Phase 10 semantic-first pipeline. Each item has a dedicated section below; the overview tracks what landed and what is still open.

| Item | What landed | Status |
|---|---|---|
| 1. Retrieval comparison + clarifying-question flow | Hybrid (semantic + BM25/RRF) retrieval path with an A/B against semantic; `clarify.py` vague-question gate | Landed — see "Hybrid Search A/B" and "Clarifying-Question Flow" below |
| 2. Page-number citations | `page` field flows retrieval → `SourceInfo`/`SourceItem` → `citations.py` rendering | Landed (mu-side) — pending Team Lambda's ingestion fix to actually emit `page` |
| 3. Conversation quality | `POST /conversations/summarize` + history compaction (older turns summarized instead of hard-dropped) | Landed — see "Conversation Quality" below |
| 4. Eval suite expansion | Main suite 15 → 35 cases, new `must_ask_clarification` expectation, targeted 18-case regression runner | Landed — see "Eval suite expansion" below |
| 5. Flashcard–chatbot connection | Feasibility-only investigation (stateless design; no Team Lambda flashcard API exists to query) | Parked — needs a scoped ask to Team Lambda first |

### Eval suite expansion (item 4)

- `eval/cases.json` grew from the **15 shared cases** used in the hybrid A/B to **35 cases**, including **7 clarification cases** and **5 multi-hop cases** (3 of the multi-hop cases are new).
- Case expectations now support a **`must_ask_clarification`** field: `eval/eval_suite.py` asserts `result.needed_clarification` matches it (the full pipeline diff vs `ask()`), and `eval/run_regression.py` skips clarification cases since they are covered by the unit suite.
- A targeted **`eval/cases_regression.json`** (18 cases = 15 original Phase 7 cases + 3 new multi-hop cases) backs `python eval/run_regression.py` — a throttled, faster check against real Groq for the non-clarification subset.
- Latest full-suite run: `eval/eval_report.md` — **31/33 PASS** (2 failures were clarification cases later addressed by the heuristic widening below).

### Clarification heuristic widening

The vague-question detector (`clarify.py`) flags a question as vague when it has no informative content tokens. The `NON_INFORMATIVE` term set in `bm25.py` was widened with **`explain further go else`** so that discourse-only follow-ups like "go on", "explain further", or "what else?" are treated as content-free and correctly prompt for clarification. This is the fix behind the two `must_ask_clarification` cases that initially failed in `eval/eval_report.md`.

---

## Retrieval Strategy (Phase 10)

Our retrieval pipeline uses a **semantic-first approach** with **LLM re-ranking**.

### 1. Semantic Search
We use `all-MiniLM-L6-v2` to embed questions and retrieve the top 10 candidates from ChromaDB. Our diagnostic tests (Phase 10 Part B) confirmed that semantic search is already highly effective at capturing exact terms, acronyms, and numbers — even with rephrased queries (see `scripts/diagnose_retrieval_v2.py`). This made keyword-based hybrid search unnecessary for our current scope.

### 2. LLM Re-ranking
We use an LLM call to pick the top 3-4 most relevant chunks from the 10 candidates. The re-ranking prompt instructs the LLM to prioritize chunks that cover different aspects of a question or reveal contradictions.

**Important finding from multi-run testing (5 runs x 2 prompts):** For this specific tested conflict scenario (temperature contradiction), the Groq model (GPT-OSS-120B) already included both conflicting chunks in 100% of runs, even with the simpler "relevance only" prompt. This indicates the base model handles this specific type of two-source numeric conflict reasonably well; the "conflict-aware" prompt is added as explicit reinforcement and documentation of architectural intent, rather than a fix for a previously reproducible failure in this exact scenario. Its broader value lies in:
- Guarding against potential regression if the base model changes
- Providing explicit guidance for more ambiguous or subtle conflict scenarios
- Documenting architectural intent clearly in the prompt itself

### Rationale for choosing Re-ranking over Hybrid Search
We considered adding keyword-based hybrid search to catch "missed" exact terms. However, our diagnostic tests showed that semantic search was already retrieving those terms correctly. The primary failures we observed (negation being ignored, conflicts being missed) were issues of **LLM comprehension and heuristic scope**, not initial retrieval gaps. Enhancing the LLM re-ranking step to be more "aware" of multi-source contexts was deemed a higher-impact improvement — primarily as a safeguard and explicit instruction layer.

---

## Hybrid Search A/B (Phase 11 Task 1)

Phase 11 added a **hybrid retrieval path** (semantic + BM25 fused via reciprocal rank fusion) alongside the existing semantic-only path, so both can be compared on the same question set while holding LLM re-ranking constant.

### Architecture

- **`bm25.py`**: Pure-Python Okapi BM25 index (no external dependencies). Tokenizes on lowercase alphanumeric, removes a small English stopword set + content-free terms. Builds a static index over the collection's documents via `collection.get()`.
- **`hybrid_search.py`**: Fuses ChromaDB semantic search (`all-MiniLM-L6-v2` top-20) with BM25 lexical top-20 using **Reciprocal Rank Fusion** (k=60). Returns the same dict shape as `vector_store.retrieve()` so downstream LLM re-ranking, grounding, and merging are unchanged.
- **Toggle**: `RETRIEVAL_METHOD=semantic|hybrid` env var (default `semantic`). Also available as `retrieval_method=` kwarg through `ask()`, `prepare_ask()`, `_retrieve_round()`.
- **Hybrid relevance gate**: `is_hybrid_relevant()` accepts when (a) any fused chunk has L2 distance <= max_distance, OR (b) at least one fused chunk is lexical-only (real keyword overlap — off-topic queries share no content words with the corpus, so BM25 contributes nothing).
- **Lexical index caching**: Built lazily on first hybrid query, cached on the engine keyed by sorted document id set. Invalidated automatically when new documents are added. First-call overhead: ~10 ms on demo corpus.

### Comparison Results (15 shared eval cases)

**Retrieval-level** (top-10 candidate pool, no LLM reranking):

| metric | semantic | hybrid |
|---|---|---|
| mean anchor hit@4 | 0.98 | 0.96 |
| mean anchor hit@10 | 1.00 | 1.00 |
| mean prefix hit@4 | 1.00 | 1.00 |
| mean prefix hit@10 | 1.00 | 1.00 |
| steady-state latency (ms) | ~65 | ~70 |
| gate refusals | 4 | 3 |

Per-case anchor winner: **13 ties, 0 wins either way** (2 refusal cases have no anchors).

**Notable gate difference:** `followup_conflict_reference` ("Which one was wrong?") — semantic gate refuses, hybrid gate passes. BM25 catches keyword overlap that semantic embedding distance alone misses on vague follow-ups. In the live pipeline, the rewrite step resolves this for both methods, so it is not a real answer-quality gap — but it demonstrates that BM25 provides a slightly more permissive first-pass gate on referential queries.

**Answer-level** (full pipeline, rerank=False, 5-case clean sample — accepted as final):

| method | passed | grounded |
|---|---|---|
| semantic | 5/5 | 5 |
| hybrid | 5/5 | 5 |

Parity was confirmed on a clean 5-case sample (5/5 PASS, 5/5 grounded for **both** methods). The full 15-case answer-level run (30 `ask()` calls, ~90 sequential LLM calls: rewrite/answer/grounding per case) was attempted **three times**, each a few hours apart, and every attempt was blocked by Groq free-tier rate limits. Even with proactive throttling (~3s spacing, ~20 RPM ceiling) and fallback retry, the API returned "busy right now" on essentially every non-refusal case. This is a **structural free-tier limit** (~30 RPM ceiling cannot sustain ~90 sequential calls), not a transient deployment issue. The answer-level comparison is documented for future re-run should the API tier change, via `python scripts/compare_hybrid_vs_semantic.py --with-answers` (proactive throttle + fallback retry are built into the script).

### Recommendation

**Keep semantic-only as the default; hybrid is available as an optional path but does not justify a default switch on the current corpus.**

Rationale:
1. Retrieval quality is statistically indistinguishable — the demo corpus is small (7 chunks) and semantically diverse enough that MiniLM embeddings already retrieve all relevant content at top-10. This was consistently observed across the retrieval-level evaluation (15 cases, all ties on anchor coverage) and the accepted 5-case answer-level sample (5/5 pass, 5/5 grounded for both).
2. Latency is equivalent (~65-70 ms steady-state for both retrieval methods).
3. BM25 hybrid adds complexity (index building, caching, RRF fusion, a hybrid relevance gate) with no measurable answer-quality gain on this corpus. The one gate-behavior difference (hybrid lets through a vague follow-up that semantic refuses) is mitigated by the existing query-rewrite step.
4. Hybrid may become valuable as the corpus grows with documents where semantic similarity fails to capture exact terms (e.g., code, acronyms, part numbers) — the toggle is already wired for future re-evaluation. To re-run: `python scripts/compare_hybrid_vs_semantic.py --with-answers`.

The comparison script (`scripts/compare_hybrid_vs_semantic.py`) and the final detailed report (`eval/hybrid_vs_semantic_report.md`) are kept for future re-evaluation as the corpus evolves.

---

## Clarifying-Question Flow (Phase 11 Task 1)

When a user asks a question that is too vague to retrieve or answer meaningfully (e.g., "tell me more", "what about that?", pronoun-only follow-ups with no resolvable referent), the chatbot now responds with a **clarifying question** instead of guessing or retrieving blindly.

### Design

- **`clarify.py`**: Lightweight heuristic detection — no ML model. A question is flagged as vague when it contains zero content tokens (stopwords and discourse terms filtered out) **and** the conversation history does not provide enough concrete context to resolve the referent.
- **Short-circuit**: Runs before query rewriting and retrieval in `prepare_ask()`. If clarification is required, the pipeline returns a `PreparedAsk` with `clarification_required=True` and a templated message, skipping all LLM and retrieval calls.
- **Conservative bias**: Only triggers on clearly substance-free queries. A short-but-clear question like "what is ATP?" has a concrete subject and passes through. A history-resolvable follow-up like "what about that?" after a substantive assistant reply also passes through.

### Test Coverage

| test | scenario | expected |
|---|---|---|
| `test_vague_question_triggers_clarification` | "tell me more", "what about that?" with no history | triggers clarification |
| `test_short_but_clear_question_no_clarification` | "What is photosynthesis?" | no clarification |
| `test_followup_resolvable_from_history_no_clarification` | "what about that?" after a rich assistant answer | no clarification |
| `test_prepare_ask_returns_clarification_for_vague_question` | prepare_ask integration | clarification_required=True |
| `test_ask_returns_clarifying_message_without_llm` | full ask() pipeline | needed_clarification=True, correct message |
| `test_ask_clarification_returns_clarifying_message` | /ask endpoint mock | CLARIFICATION_MESSAGE in response |
| `test_ask_stream_clarification_metadata_done` | /ask/stream endpoint mock | metadata with is_clarification=True |

> **Widened heuristic (Phase 11 item 4):** `NON_INFORMATIVE` in `bm25.py` additionally treats `explain / further / go / else` as non-content, so follow-ups like "go on", "explain further", and "what else?" are now flagged vague. See "Clarification heuristic widening" in the Phase 11 overview above.

---

## Source Attribution: Page Numbers in Citations (Phase 11 Task 2)

Page-number support is implemented on the **citation/display side** of the rag-engine, **pending real page metadata from Team Lambda's ingestion fix** (ai-ml/ingestion/pdf/). That dependency is noted but not yet resolved — ingestion currently ships chunks without a `page` field.

### Field name / schema

The consuming side reads an optional metadata key **`page`** (integer) from a retrieved chunk's metadata, matching what Team Lambda's ingestion fix is expected to emit. It flows through the existing chunk-metadata structures:

| structure | file | field |
|---|---|---|
| retrieval metadata dict | ChromaDB chunk metadata (Lambda side) | `"page": int` — optional/nullable |
| `SourceInfo` dataclass | `rag_service.py` | `page: int | None = None` |
| `SourceItem` schema | `schemas.py` | `page: int | None = None` |

`SourceInfo`/`SourceItem` already carried `document` + `source`; `page` is additive, so old chunks (ingested before Lambda's fix ships) simply omit it. `_build_sources()` tolerantly coerces the metadata value to `int` and falls back to `None` on missing/unparseable values. The value survives the cache round-trip (`cache.py` / `main.py` `_cache_entry_to_result`).

### Display

`citations.py` provides `format_citation(source)`, which renders:

- `Source: [Document Name], Page 4` — when `page` is present
- `Source: [Document Name]` — when `page` is missing (no `Page None` / broken text)

Both CLI demos (`chunked_retrieval.py`, `memory/conversational_rag.py`) use it. When Team Lambda's ingestion starts emitting `page`, no code change is needed on this side.

### Test coverage

`tests/test_citations.py` verifies with mocked chunk data: one chunk with `page` present and one without — both render correctly — plus fallback behavior with no document/page and `_build_sources()` reading `page` from mocked retrieval metadata.

---

## Conversation Quality (Phase 11 Part C)

### Investigation finding: history is truncated by design

History has always been capped to the last **4 user/assistant turns** (8 messages) before the LLM sees it:

- `rag_service.py:HISTORY_TURN_CAP = 4` — applied via `recent_history()` in `rewrite_question()` and `build_messages()`.
- Every call is **stateless** — the client sends the full `history` list with each `/ask` request (there is no server-side session store; `_append_history()` is effectively dead code in the API layer).
- This means facts from earlier turns are silently dropped, not just faded: by turn 6+, early-turn context is structurally invisible to the model.

### New endpoint: POST /conversations/summarize

Added `POST /conversations/summarize` (request: `SummarizeRequest { history }`, response: `SummarizeResponse { summary, turn_count, cached }`).

- Uses a **dedicated summarization prompt** in `rag_service.py` (`SUMMARY_SYSTEM_PROMPT`), separate from the RAG answer prompt and involving no retrieval.
- In-memory LRU+TTL summary cache (`cache.py:SummaryCache`) keyed on `(user, history)` — repeat calls return the cached summary without hitting the LLM.
- Endpoint is authenticated and rate-limited.

### Compaction fix: summary of older turns instead of hard drop

Instead of silently dropping turns older than the cap, `recent_history()` now optionally condenses them into a single summary block:

- **`condense_history(client, history, max_turns=4)`**: sends the older turns to `summarize_conversation()`, returns `[summary_block] + last 4 turns verbatim`. Returns `None` on failure, so the caller falls back to the hard-drop cap.
- **`recent_history(history, max_turns=4, client=None)`**: when a client is provided and the history is overdue, condenses instead of dropping. If a summary block is already present, it is preserved and only the verbatim tail is truncated.
- **Single LLM call per request**: compaction happens once in `prepare_ask()` (before rewrite + answer generation), not duplicated in `rewrite_question()` or `build_messages()`.
- **`ENABLE_HISTORY_COMPACTION` env var** (default `true`): allows disabling compaction for baseline regression tests. When `false`, falls back to the original hard-drop cap.
- **Graceful fallback**: if the summarization LLM call fails, the cap falls back to the original hard-drop behavior.

### Baseline degradation test (unit + integration)

- `tests/test_conversation_summary.py` — 16 unit tests covering `summarize_conversation()`, `condense_history()`, `recent_history()` (plain cap, compaction, toggle, summary-preservation), endpoint contract, and a token-savings assertion (compacted context < full raw history while retaining early-turn facts).
- `tests/test_conversation_quality_integration.py` — opt-in live-server tests (`RUN_LIVE_CONVERSATION_TESTS=1`) against real Groq, no mocks. Runs a 12-turn conversation script with probe questions referencing turn 1, for both baseline (no compaction) and compaction-enabled servers. Asserts: baseline does NOT recall the early-turn token; compaction DOES recall it. Also exercises `/conversations/summarize` live and measures token savings vs full history.

### Cache key note (flag for Maryam/TL)

`_cache_key()` in `main.py` hashes the **raw client-supplied history** — compaction happens server-side and does not change this key. However, two conversations differing only in turns older than `HISTORY_TURN_CAP` will produce **different keys but nearly identical effective context after compaction**. This is documented as a TODO: changing the key formula could improve hit-rate for long conversations but would affect invalidation/collision semantics and needs sign-off.
