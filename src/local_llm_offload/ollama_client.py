"""Thin HTTP client around a remote Ollama server.

Uses the OpenAI-compatible endpoint (`/v1/chat/completions`) for chat
completions, and Ollama's native endpoint (`/api/tags`) to list installed
models — mirroring what the two exposed MCP tools need.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx

from . import config


class OllamaError(Exception):
    """Raised whenever the Ollama server can't be reached or errors out.

    The message is meant to be shown as-is to whoever called the MCP tool,
    so it's written as a complete, human-readable sentence rather than a
    stack trace fragment.
    """


def _base_url() -> str:
    return config.OLLAMA_HOST


@contextmanager
def _translate_errors(operation: str) -> Iterator[None]:
    """Turns httpx exceptions into an `OllamaError` with an actionable message."""
    try:
        yield
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Impossibile connettersi al server Ollama su {config.OLLAMA_HOST}. "
            f"Verifica che sia in esecuzione e raggiungibile in rete. Dettagli: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Timeout ({config.OLLAMA_TIMEOUT_SECONDS}s) su {config.OLLAMA_HOST} durante {operation}. "
            f"Modelli grandi (es. qwen3-coder:30b) possono richiedere OLLAMA_TIMEOUT_SECONDS più alto."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Il server Ollama ha risposto con errore {exc.response.status_code} durante {operation}: "
            f"{exc.response.text[:500]}"
        ) from exc


async def list_models() -> list[dict]:
    """Return the list of models installed on the Ollama server (/api/tags)."""
    url = f"{_base_url()}/api/tags"
    with _translate_errors("la lista dei modelli"):
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()

    data = response.json()
    return data.get("models", [])


async def chat_completion(prompt: str, model: str, system_prompt: str | None = None) -> str:
    """Send a non-streaming chat completion request. Used by tests/simple callers."""
    url = f"{_base_url()}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "stream": False}

    with _translate_errors(f"la generazione con '{model}'"):
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OllamaError(
            f"Risposta inattesa dal server Ollama (formato OpenAI-compatible non rispettato): {data}"
        ) from exc


@dataclass
class ChatResult:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


async def chat_completion_stream(
    prompt: str,
    model: str,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    system_prompt: str | None = None,
) -> ChatResult:
    """Stream a chat completion, calling `on_chunk` with each text delta as it arrives.

    Requesting `stream_options.include_usage` gets Ollama to emit a final
    chunk carrying real prompt/completion token counts, same as the
    non-streaming response's `usage` field — needed for get_usage_stats.
    """
    url = f"{_base_url()}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.monotonic()
    text_parts: list[str] = []
    usage: dict[str, int] = {}

    with _translate_errors(f"la generazione con '{model}'"):
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: ") :]
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    for choice in chunk.get("choices") or []:
                        content = choice.get("delta", {}).get("content")
                        if content:
                            text_parts.append(content)
                            if on_chunk:
                                await on_chunk(content)

                    if chunk.get("usage"):
                        usage = chunk["usage"]

    return ChatResult(text="".join(text_parts), usage=usage, elapsed_seconds=time.monotonic() - start)
