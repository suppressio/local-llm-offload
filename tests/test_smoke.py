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
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0


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


def test_unreachable_host_raises_clear_error() -> None:
    original_host = config.OLLAMA_HOST
    config.OLLAMA_HOST = "http://127.0.0.1:1"  # porta non in ascolto
    try:
        with pytest.raises(ollama_client.OllamaError, match="Impossibile connettersi"):
            asyncio.run(ollama_client.list_models())
    finally:
        config.OLLAMA_HOST = original_host
