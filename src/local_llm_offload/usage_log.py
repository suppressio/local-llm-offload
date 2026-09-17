"""Local JSONL log of delegate_to_local_llm calls, and a summary over it.

This is the "real token savings" measurement the project set out to get
before considering any automatic delegation: every call records the actual
token counts Ollama reports, so `get_usage_stats` can answer "how much did
we actually offload" instead of guessing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class UsageEntry:
    timestamp: float
    model: str
    ok: bool
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def record(entry: UsageEntry) -> None:
    """Append one usage entry as a JSON line. Failures here never raise.

    Logging is a side effect of the real tool call; a full disk or a
    permissions issue on the log file must not turn a working
    delegate_to_local_llm call into a failed one.
    """
    try:
        config.USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
    except OSError:
        pass


def _read_entries() -> list[UsageEntry]:
    if not config.USAGE_LOG_PATH.is_file():
        return []

    entries: list[UsageEntry] = []
    with config.USAGE_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(UsageEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


@dataclass
class ModelSummary:
    calls: int = 0
    ok_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class UsageSummary:
    by_model: dict[str, ModelSummary] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return sum(m.total_tokens for m in self.by_model.values())

    @property
    def elapsed_seconds(self) -> float:
        return sum(m.elapsed_seconds for m in self.by_model.values())


def summarize(since_hours: float | None = None) -> UsageSummary:
    entries = _read_entries()
    if since_hours is not None:
        cutoff = time.time() - since_hours * 3600
        entries = [e for e in entries if e.timestamp >= cutoff]

    summary = UsageSummary()
    for entry in entries:
        model_summary = summary.by_model.setdefault(entry.model, ModelSummary())
        model_summary.calls += 1
        if entry.ok:
            model_summary.ok_calls += 1
        model_summary.prompt_tokens += entry.prompt_tokens
        model_summary.completion_tokens += entry.completion_tokens
        model_summary.total_tokens += entry.total_tokens
        model_summary.elapsed_seconds += entry.elapsed_seconds

    return summary
