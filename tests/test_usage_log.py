"""Unit tests for the usage log: pure logic, no network involved."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_llm_offload import config, usage_log


@pytest.fixture(autouse=True)
def isolated_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setattr(config, "USAGE_LOG_PATH", log_path)
    return log_path


def test_summarize_with_no_log_returns_zero_calls() -> None:
    summary = usage_log.summarize()
    assert summary.calls == 0
    assert summary.total_tokens == 0


def test_record_and_summarize_aggregates_by_model() -> None:
    usage_log.record(
        usage_log.UsageEntry(
            timestamp=time.time(), model="qwen2.5-coder:7b", ok=True, elapsed_seconds=1.0,
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
    )
    usage_log.record(
        usage_log.UsageEntry(
            timestamp=time.time(), model="qwen2.5-coder:7b", ok=False, elapsed_seconds=0.0,
        )
    )
    usage_log.record(
        usage_log.UsageEntry(
            timestamp=time.time(), model="qwen3-coder:30b", ok=True, elapsed_seconds=3.0,
            prompt_tokens=20, completion_tokens=10, total_tokens=30,
        )
    )

    summary = usage_log.summarize()

    assert summary.calls == 3
    assert summary.total_tokens == 45

    small = summary.by_model["qwen2.5-coder:7b"]
    assert small.calls == 2
    assert small.ok_calls == 1
    assert small.total_tokens == 15

    big = summary.by_model["qwen3-coder:30b"]
    assert big.calls == 1
    assert big.total_tokens == 30


def test_summarize_since_hours_excludes_old_entries() -> None:
    old_entry = usage_log.UsageEntry(
        timestamp=time.time() - 3600 * 5, model="qwen2.5-coder:7b", ok=True,
        elapsed_seconds=1.0, prompt_tokens=1, completion_tokens=1, total_tokens=2,
    )
    recent_entry = usage_log.UsageEntry(
        timestamp=time.time(), model="qwen2.5-coder:7b", ok=True,
        elapsed_seconds=1.0, prompt_tokens=3, completion_tokens=3, total_tokens=6,
    )
    usage_log.record(old_entry)
    usage_log.record(recent_entry)

    summary = usage_log.summarize(since_hours=1.0)

    assert summary.calls == 1
    assert summary.total_tokens == 6


def test_malformed_lines_are_skipped(isolated_log_path: Path) -> None:
    isolated_log_path.parent.mkdir(parents=True, exist_ok=True)
    with isolated_log_path.open("w", encoding="utf-8") as f:
        f.write("not json\n")
        f.write("\n")
        f.write('{"unexpected_field": true}\n')

    summary = usage_log.summarize()

    assert summary.calls == 0
