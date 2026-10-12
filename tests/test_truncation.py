"""Unit tests for the truncation guard: pure logic, no network involved."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_offload import config, ollama_client, server


def test_short_prompts_are_never_flagged() -> None:
    assert not ollama_client.looks_truncated(prompt_chars=500, prompt_tokens=10)


def test_missing_token_count_is_not_flagged() -> None:
    assert not ollama_client.looks_truncated(prompt_chars=100_000, prompt_tokens=0)


def test_normal_chars_per_token_ratio_is_not_flagged() -> None:
    # ~4 chars/token: ordinary prose or code that fit in the window.
    assert not ollama_client.looks_truncated(prompt_chars=40_000, prompt_tokens=10_000)


def test_heavily_truncated_prompt_is_flagged_without_knowing_the_window() -> None:
    # Measured on a real server: 108 912 chars sent, num_ctx 8192, 4098 tokens processed.
    assert ollama_client.looks_truncated(prompt_chars=108_912, prompt_tokens=4_098)


def test_slightly_overflowing_prompt_is_flagged_by_half_window() -> None:
    # Measured: ~121k chars (~33k tokens) into a 32 768 window -> 16 386 tokens processed.
    # The chars/token ratio alone (~7.4 -> borderline) isn't conclusive; the half-window mark is.
    assert ollama_client.looks_truncated(prompt_chars=99_000, prompt_tokens=16_386, context_window=32_768)


def test_prompt_far_from_half_window_is_not_flagged() -> None:
    assert not ollama_client.looks_truncated(prompt_chars=80_000, prompt_tokens=20_000, context_window=32_768)


def test_repetitive_complete_prompt_is_not_flagged_when_window_is_known() -> None:
    # Highly compressible text can exceed the fallback ratio while fitting fine.
    assert not ollama_client.looks_truncated(prompt_chars=70_000, prompt_tokens=7_000, context_window=32_768)


@pytest.fixture
def small_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_CONTEXT_FILE_CHARS", 100)
    monkeypatch.setattr(config, "MAX_TOTAL_CONTEXT_CHARS", 150)


def test_files_within_caps_are_not_reported(tmp_path: Path, small_caps: None) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x" * 50)
    block, cut = server._load_context_files([str(f)])
    assert "x" * 50 in block
    assert cut == []


def test_file_over_per_file_cap_is_reported(tmp_path: Path, small_caps: None) -> None:
    f = tmp_path / "big.txt"
    f.write_text("x" * 120)
    _, cut = server._load_context_files([str(f)])
    assert cut == [str(f)]


def test_total_cap_reports_cut_file_and_skipped_ones(tmp_path: Path, small_caps: None) -> None:
    paths = []
    for name in ("a", "b", "c"):
        f = tmp_path / f"{name}.txt"
        f.write_text("x" * 90)
        paths.append(str(f))
    _, cut = server._load_context_files(paths)
    # a fits (90), b overflows the 150 total and is cut, c is never read.
    assert cut == paths[1:]


def test_no_warning_when_input_was_complete() -> None:
    result = ollama_client.ChatResult(text="ok", usage={"prompt_tokens": 100})
    assert server._truncation_warnings([], result) == ""


def test_warning_mentions_ollama_truncation() -> None:
    result = ollama_client.ChatResult(text="ok", usage={"prompt_tokens": 4098}, truncated=True, context_window=8192)
    warning = server._truncation_warnings([], result)
    assert warning.startswith("[local-llm-offload] ATTENZIONE")
    assert "4098" in warning
    assert "8192" in warning
    assert "OLLAMA_NUM_CTX" in warning


def test_warning_lists_cut_files() -> None:
    result = ollama_client.ChatResult(text="ok", usage={"prompt_tokens": 100})
    warning = server._truncation_warnings(["/tmp/a.log"], result)
    assert "/tmp/a.log" in warning
    assert "MAX_CONTEXT_FILE_CHARS" in warning
