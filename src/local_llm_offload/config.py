"""Runtime configuration read from environment variables.

No .env auto-loading here on purpose: when registered as an MCP server via
`claude mcp add ... --env KEY=VALUE`, Claude Code injects the process
environment directly. A .env.example is provided in the repo root purely as
documentation of the variables below.
"""

from __future__ import annotations

import os


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

# Per-file and total character caps applied to `context_files`, to avoid
# blowing past the local model's context window with a single large file.
MAX_CONTEXT_FILE_CHARS: int = _get_int("MAX_CONTEXT_FILE_CHARS", 20_000)
MAX_TOTAL_CONTEXT_CHARS: int = _get_int("MAX_TOTAL_CONTEXT_CHARS", 60_000)
