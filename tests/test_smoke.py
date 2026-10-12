"""Smoke tests against a real Ollama server.

These are not mocked on purpose: the whole point of this bridge is that it
talks to a real Ollama instance, so the smoke test exercises exactly that.
It skips (rather than fails) when no server is reachable, so it doesn't
break CI or a machine without Ollama configured.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from local_llm_offload import config, ollama_client


def _ollama_reachable() -> bool:
    try:
        response = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=(
        f"Nessun server Ollama raggiungibile su {config.OLLAMA_HOST}. "
        "Imposta OLLAMA_HOST per eseguire questo smoke test."
    ),
)


@requires_ollama
def test_list_models_returns_at_least_one_model() -> None:
    models = asyncio.run(ollama_client.list_models())
    assert isinstance(models, list)
    assert len(models) > 0
    assert "name" in models[0]


@requires_ollama
def test_chat_completion_returns_nonempty_text() -> None:
    reply = asyncio.run(
        ollama_client.chat_completion(
            "Rispondi con una sola parola: ciao",
            model=config.OLLAMA_DEFAULT_MODEL,
        )
    )
    assert len(reply.text.strip()) > 0
    assert reply.usage["total_tokens"] > 0
    assert not reply.truncated


@requires_ollama
def test_chat_completion_stream_reports_chunks_and_usage() -> None:
    received: list[str] = []

    async def on_chunk(text: str) -> None:
        received.append(text)

    result = asyncio.run(
        ollama_client.chat_completion_stream(
            "Conta da uno a cinque, una cifra per riga.",
            model=config.OLLAMA_DEFAULT_MODEL,
            on_chunk=on_chunk,
        )
    )
    assert len(result.text.strip()) > 0
    assert result.text == "".join(received)
    assert result.usage.get("total_tokens", 0) > 0
    assert result.elapsed_seconds > 0


def _long_prompt(lines: int) -> str:
    # Prose-like lines (~4 chars/token), so the token count is realistic.
    body = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(lines))
    return f"{body}\n\nWhat is the number of the last line above? Reply with the number only."


@requires_ollama
def test_long_prompt_fits_when_num_ctx_is_large(monkeypatch: pytest.MonkeyPatch) -> None:
    # ~30k tokens against a 32k window: must reach the model in full.
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 32_768)
    result = asyncio.run(ollama_client.chat_completion(_long_prompt(1_900), model=config.OLLAMA_DEFAULT_MODEL))
    if result.context_window < 32_768:
        pytest.skip(f"{config.OLLAMA_DEFAULT_MODEL} supporta solo {result.context_window} token di contesto")
    assert result.usage["prompt_tokens"] > 25_000
    assert not result.truncated


@requires_ollama
def test_long_prompt_is_flagged_when_num_ctx_is_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 4_096)
    result = asyncio.run(ollama_client.chat_completion(_long_prompt(1_900), model=config.OLLAMA_DEFAULT_MODEL))
    assert result.truncated


@requires_ollama
def test_think_level_is_ignored_by_model_without_thinking() -> None:
    info = asyncio.run(ollama_client.model_info(config.OLLAMA_DEFAULT_MODEL))
    if info is None or info.can_think:
        pytest.skip(f"{config.OLLAMA_DEFAULT_MODEL} supporta il ragionamento: test non applicabile")
    # Sent as-is, Ollama would answer HTTP 400 "does not support thinking".
    result = asyncio.run(
        ollama_client.chat_completion("Rispondi con una sola parola: ciao", model=config.OLLAMA_DEFAULT_MODEL, think="low")
    )
    assert len(result.text.strip()) > 0


def _thinking_model() -> str | None:
    # Smallest first: loading a big model just for this check would be slow.
    models = sorted(asyncio.run(ollama_client.list_models()), key=lambda m: m.get("size", 0))
    for entry in models:
        info = asyncio.run(ollama_client.model_info(entry["name"]))
        if info and info.can_think:
            return entry["name"]
    return None


@requires_ollama
def test_think_off_is_accepted_by_thinking_model() -> None:
    model = _thinking_model()
    if model is None:
        pytest.skip("Nessun modello con capacità di ragionamento installato")
    result = asyncio.run(ollama_client.chat_completion("Rispondi con una sola parola: ciao", model=model, think="off"))
    assert len(result.text.strip()) > 0


def test_unreachable_host_raises_clear_error() -> None:
    original_host = config.OLLAMA_HOST
    config.OLLAMA_HOST = "http://127.0.0.1:1"  # porta non in ascolto
    try:
        with pytest.raises(ollama_client.OllamaError, match="Impossibile connettersi"):
            asyncio.run(ollama_client.list_models())
    finally:
        config.OLLAMA_HOST = original_host
