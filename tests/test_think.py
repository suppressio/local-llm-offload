"""Unit tests for the reasoning-level option: pure logic, no network involved."""

from __future__ import annotations

import asyncio

import pytest

from local_llm_offload import config, ollama_client


@pytest.fixture(autouse=True)
def known_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pre-filled cache, so no /api/show request is made.
    monkeypatch.setattr(
        ollama_client,
        "_model_info",
        {
            "thinker": ollama_client.ModelInfo(max_context=32_768, can_think=True),
            "plain": ollama_client.ModelInfo(max_context=32_768, can_think=False),
        },
    )
    monkeypatch.setattr(config, "OLLAMA_THINK", "")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("  ", None), ("off", False), ("ON", True), ("low", "low"), (" High ", "high")],
)
def test_parse_think_maps_levels(value: str | None, expected: bool | str | None) -> None:
    assert ollama_client.parse_think(value) == expected


def test_parse_think_rejects_unknown_values() -> None:
    with pytest.raises(ollama_client.OllamaError, match="think non valido"):
        ollama_client.parse_think("a bit")


def _payload(model: str, think: str | None) -> dict:
    return asyncio.run(ollama_client._chat_payload("hi", model, None, stream=False, think=think))


def test_no_level_means_no_think_field() -> None:
    assert "think" not in _payload("thinker", None)


def test_per_call_level_is_sent() -> None:
    assert _payload("thinker", "low")["think"] == "low"


def test_env_default_is_used_when_call_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_THINK", "off")
    assert _payload("thinker", None)["think"] is False


def test_per_call_level_wins_over_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_THINK", "off")
    assert _payload("thinker", "medium")["think"] == "medium"


def test_explicitly_empty_call_value_overrides_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_THINK", "off")
    assert "think" not in _payload("thinker", "")


def test_level_is_dropped_for_models_that_cannot_think() -> None:
    assert "think" not in _payload("plain", "low")


def test_unknown_capabilities_only_send_off(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unreadable(model: str) -> None:
        return None

    monkeypatch.setattr(ollama_client, "model_info", unreadable)
    assert _payload("unknown", "off")["think"] is False
    assert "think" not in _payload("unknown", "low")
