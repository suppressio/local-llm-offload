"""Runtime configuration read from environment variables.

No .env auto-loading here on purpose: when registered as an MCP server via
`claude mcp add ... --env KEY=VALUE`, Claude Code injects the process
environment directly. A .env.example is provided in the repo root purely as
documentation of the variables below.
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_DEFAULT_MODEL: str = os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT_SECONDS: float = _get_float("OLLAMA_TIMEOUT_SECONDS", 120.0)

# Context window (tokens) requested from Ollama on every call via
# `options.num_ctx`. Without it Ollama falls back to the server default
# (small unless OLLAMA_CONTEXT_LENGTH is set there) and silently drops
# whatever doesn't fit. Kept fixed rather than sized per request because
# Ollama reloads the model whenever num_ctx changes. Bigger values cost
# VRAM/RAM for the KV cache; Ollama also caps it at the model's own maximum.
OLLAMA_NUM_CTX: int = _get_int("OLLAMA_NUM_CTX", 16_384)

# Per-file and total character caps applied to `context_files`, to avoid
# blowing past the local model's context window with a single large file.
# The total defaults to ~2 chars per token of OLLAMA_NUM_CTX: at the usual
# 3-4 chars/token of code and prose that fills about half the window,
# leaving room for the prompt and the answer.
MAX_TOTAL_CONTEXT_CHARS: int = _get_int("MAX_TOTAL_CONTEXT_CHARS", OLLAMA_NUM_CTX * 2)
MAX_CONTEXT_FILE_CHARS: int = _get_int("MAX_CONTEXT_FILE_CHARS", min(20_000, MAX_TOTAL_CONTEXT_CHARS))

# JSONL log of every delegate_to_local_llm call (model, tokens, elapsed time),
# used by get_usage_stats to report real token/time savings. Path.home() is
# used instead of an XDG-style path so the default works the same way on
# Windows and Unix without extra dependencies.
USAGE_LOG_PATH: Path = Path(
    os.environ.get("USAGE_LOG_PATH", str(Path.home() / ".local-llm-offload" / "usage.jsonl"))
).expanduser()

# How many streamed chunks to batch before sending one MCP progress
# notification, to avoid flooding the client with one message per token.
PROGRESS_CHUNK_INTERVAL: int = _get_int("PROGRESS_CHUNK_INTERVAL", 20)
