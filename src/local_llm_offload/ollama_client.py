"""Thin HTTP client around a remote Ollama server.

Uses the OpenAI-compatible endpoint (`/v1/chat/completions`) for chat
completions, and Ollama's native endpoint (`/api/tags`) to list installed
models — mirroring what the two exposed MCP tools need.
"""

from __future__ import annotations

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


async def list_models() -> list[dict]:
    """Return the list of models installed on the Ollama server (/api/tags)."""
    url = f"{_base_url()}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Impossibile connettersi al server Ollama su {config.OLLAMA_HOST}. "
            f"Verifica che sia in esecuzione e raggiungibile in rete. Dettagli: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Timeout ({config.OLLAMA_TIMEOUT_SECONDS}s) contattando {config.OLLAMA_HOST}/api/tags."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Il server Ollama ha risposto con errore {exc.response.status_code} "
            f"su {url}: {exc.response.text[:500]}"
        ) from exc

    data = response.json()
    return data.get("models", [])


async def chat_completion(prompt: str, model: str, system_prompt: str | None = None) -> str:
    """Send a chat completion request to Ollama's OpenAI-compatible endpoint."""
    url = f"{_base_url()}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "stream": False}

    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Impossibile connettersi al server Ollama su {config.OLLAMA_HOST}. "
            f"Verifica che sia in esecuzione e raggiungibile in rete. Dettagli: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Timeout ({config.OLLAMA_TIMEOUT_SECONDS}s) in attesa di risposta dal modello "
            f"'{model}' su {config.OLLAMA_HOST}. Modelli grandi (es. qwen3-coder:30b) possono "
            f"richiedere OLLAMA_TIMEOUT_SECONDS più alto."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Il server Ollama ha risposto con errore {exc.response.status_code} "
            f"per il modello '{model}': {exc.response.text[:500]}"
        ) from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OllamaError(
            f"Risposta inattesa dal server Ollama (formato OpenAI-compatible non rispettato): {data}"
        ) from exc
