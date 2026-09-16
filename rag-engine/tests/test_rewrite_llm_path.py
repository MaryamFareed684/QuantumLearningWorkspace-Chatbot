"""Unit tests for the LLM rewriting path of rewrite_question().

The short-circuit paths (empty history, client=None) are covered in
test_rewrite_grounding.py. These tests exercise the real rewrite path with a
mocked Groq client, covering successful rewrites, fallback on empty/malformed
model output, output cleanup, and the constructed prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import GROQ_MODEL, rewrite_question  # noqa: E402

NONEMPTY_HISTORY = [
    {"role": "user", "content": "What is the Calvin cycle?"},
    {
        "role": "assistant",
        "content": "The Calvin cycle fixes carbon dioxide into sugar in the stroma.",
    },
]


def _fake_groq(text: str):
    """A minimal Groq-shaped mock that records the kwargs it was called with."""
    captured = {}

    def create(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    completions = SimpleNamespace(create=create, captured=captured)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_rewrite_successful_followup_with_mocked_llm():
    client = _fake_groq("Where does the Calvin cycle occur in the stroma?")
    rewritten = rewrite_question(
        client, list(NONEMPTY_HISTORY), "Where does it happen?"
    )
    assert rewritten == "Where does the Calvin cycle occur in the stroma?"


def test_rewrite_empty_response_falls_back_to_original():
    client = _fake_groq("")
    rewritten = rewrite_question(
        client, list(NONEMPTY_HISTORY), "Where does it happen?"
    )
    assert rewritten == "Where does it happen?"


def test_rewrite_whitespace_only_response_falls_back_to_original():
    client = _fake_groq("   \n \t  \n")
    rewritten = rewrite_question(
        client, list(NONEMPTY_HISTORY), "Where does it happen?"
    )
    assert rewritten == "Where does it happen?"


def test_rewrite_cleans_quote_wrapped_and_padded_response():
    client = _fake_groq('  "ATP synthase proton gradient"  ')
    rewritten = rewrite_question(
        client, list(NONEMPTY_HISTORY), "What enzyme makes ATP?"
    )
    assert rewritten == "ATP synthase proton gradient"


def test_rewrite_takes_first_cleaned_line_only():
    client = _fake_groq(
        '"Where does the Calvin cycle occur in the stroma"\n(extra note)'
    )
    rewritten = rewrite_question(
        client, list(NONEMPTY_HISTORY), "Where does it happen?"
    )
    assert rewritten == "Where does the Calvin cycle occur in the stroma"


def test_rewrite_prompt_includes_history_and_latest_question():
    client = _fake_groq("rewritten query")
    rewrite_question(client, list(NONEMPTY_HISTORY), "Where does it happen?")

    kwargs = client.chat.completions.captured["kwargs"]
    messages = kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert "Rewrite the user's latest question" in messages[0]["content"]

    body = messages[1]["content"]
    assert "Conversation so far:" in body
    assert "user: What is the Calvin cycle?" in body
    assert "assistant: The Calvin cycle fixes carbon dioxide" in body
    assert "Latest question: Where does it happen?" in body
    assert "Standalone search query:" in body

    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == GROQ_MODEL