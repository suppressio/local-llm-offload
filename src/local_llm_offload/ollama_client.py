"""Thin HTTP client around a remote Ollama server.

Uses Ollama's native endpoints: `/api/chat` for chat completions and
`/api/tags` to list installed models. The native chat endpoint is required
(rather than the OpenAI-compatible one) because it's the only one that
accepts `options.num_ctx`, i.e. lets this client choose the context window
instead of inheriting the server's default and having prompts truncated.
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


# How Ollama truncates: when a prompt doesn't fit the effective context
# window (num_ctx, capped at the model's own maximum) it keeps only about
# half of it, so the reported prompt_eval_count lands right at window/2 + a
# couple of tokens (measured: 4098 for 8192, 16386 for 32768). Prompt
# caching doesn't hide this: a cache hit still reports the full count.
# A complete prompt landing in this narrow band by chance is possible but
# rare (16 tokens out of thousands), and only costs a spurious warning.
_TRUNCATED_HALF_WINDOW_SLACK = 16

# Fallback when the model's maximum context is unknown: no real text gets
# this many characters per reported token unless part of it was dropped.
_FALLBACK_CHARS_PER_TOKEN = 6.0

# Below this size a prompt can't overflow any realistic context window.
_TRUNCATION_MIN_CHARS = 2_000


def looks_truncated(prompt_chars: int, prompt_tokens: int, context_window: int | None = None) -> bool:
    """Whether Ollama most likely discarded part of a prompt of `prompt_chars` characters.

    `context_window` is the effective window (requested num_ctx capped at the
    model's maximum); when known it enables the precise half-window check.
    """
    if prompt_chars < _TRUNCATION_MIN_CHARS or prompt_tokens <= 0:
        return False
    if prompt_chars / prompt_tokens > _FALLBACK_CHARS_PER_TOKEN:
        return True
    if context_window:
        half = context_window // 2
        return half <= prompt_tokens <= half + _TRUNCATED_HALF_WINDOW_SLACK
    return False


_model_max_context: dict[str, int | None] = {}


async def model_max_context(model: str) -> int | None:
    """The model's own maximum context length (/api/show), cached per process; None if unknown.

    Only answers from the server are cached: after a network error the next
    call asks again instead of disabling the half-window check for good.
    """
    if model in _model_max_context:
        return _model_max_context[model]

    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{_base_url()}/api/show", json={"model": model})
            response.raise_for_status()
        info = response.json().get("model_info") or {}
    except (httpx.HTTPError, ValueError):
        # Only feeds the truncation check: never fail the actual request over it.
        return None

    lengths = [v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int)]
    _model_max_context[model] = lengths[0] if lengths else None
    return _model_max_context[model]


async def _context_window(model: str) -> int:
    """Effective window for `model`: the requested num_ctx, capped at the model's maximum."""
    model_max = await model_max_context(model)
    return min(config.OLLAMA_NUM_CTX, model_max) if model_max else config.OLLAMA_NUM_CTX


def _chat_payload(prompt: str, model: str, system_prompt: str | None, stream: bool) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX},
    }


def _usage_from(data: dict) -> dict[str, int]:
    """Map Ollama's native counters onto the OpenAI-style keys the usage log uses."""
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@dataclass
class ChatResult:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # True when Ollama most likely cut the prompt to fit the context window.
    truncated: bool = False
    # Effective context window used (num_ctx capped at the model's maximum).
    context_window: int = 0


async def _result(payload: dict, model: str, text: str, usage: dict[str, int], start: float) -> ChatResult:
    elapsed = time.monotonic() - start
    window = await _context_window(model)
    prompt_chars = sum(len(m["content"]) for m in payload["messages"])
    return ChatResult(
        text=text,
        usage=usage,
        elapsed_seconds=elapsed,
        truncated=looks_truncated(prompt_chars, usage.get("prompt_tokens", 0), window),
        context_window=window,
    )


async def chat_completion(prompt: str, model: str, system_prompt: str | None = None) -> ChatResult:
    """Send a non-streaming chat request (native /api/chat). Used by tests/simple callers."""
    url = f"{_base_url()}/api/chat"
    payload = _chat_payload(prompt, model, system_prompt, stream=False)

    start = time.monotonic()
    with _translate_errors(f"la generazione con '{model}'"):
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

    data = response.json()
    try:
        text = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise OllamaError(f"Risposta inattesa dal server Ollama (formato /api/chat non rispettato): {data}") from exc

    usage = _usage_from(data)
    return await _result(payload, model, text, usage, start)


async def chat_completion_stream(
    prompt: str,
    model: str,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    system_prompt: str | None = None,
) -> ChatResult:
    """Stream a chat completion, calling `on_chunk` with each text delta as it arrives.

    Uses Ollama's native /api/chat, which streams one JSON object per line;
    the final one (`done: true`) carries the real prompt/completion token
    counts needed for get_usage_stats and for the truncation check.
    """
    url = f"{_base_url()}/api/chat"
    payload = _chat_payload(prompt, model, system_prompt, stream=True)

    start = time.monotonic()
    text_parts: list[str] = []
    usage: dict[str, int] = {}

    with _translate_errors(f"la generazione con '{model}'"):
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("error"):
                        raise OllamaError(f"Il server Ollama ha interrotto la generazione con '{model}': {chunk['error']}")

                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        text_parts.append(content)
                        if on_chunk:
                            await on_chunk(content)

                    if chunk.get("done"):
                        usage = _usage_from(chunk)
                        break

    return await _result(payload, model, "".join(text_parts), usage, start)
