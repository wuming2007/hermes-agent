"""Defaults for the new ``cognition`` config block (PR1)."""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG


def test_cognition_block_present():
    assert "cognition" in DEFAULT_CONFIG
    cognition = DEFAULT_CONFIG["cognition"]
    assert isinstance(cognition, dict)


def test_cognition_disabled_by_default():
    assert DEFAULT_CONFIG["cognition"]["enabled"] is False


def test_cognition_default_mode_is_standard():
    assert DEFAULT_CONFIG["cognition"]["default_mode"] == "standard"


def test_cognition_fast_mode_thresholds_present():
    fast = DEFAULT_CONFIG["cognition"]["fast_mode"]
    assert fast["max_chars"] == 160
    assert fast["max_words"] == 28
    assert fast["allow_urls"] is False
    assert fast["allow_code_blocks"] is False


def test_cognition_deep_mode_triggers_present():
    triggers = DEFAULT_CONFIG["cognition"]["deep_mode_triggers"]
    for key in (
        "historical_questions",
        "code_changes",
        "risky_external_actions",
        "architecture_decisions",
    ):
        assert key in triggers
        assert triggers[key] is True


def test_cognition_consistency_guard_defaults():
    guard = DEFAULT_CONFIG["cognition"]["consistency_guard"]
    assert guard["enabled"] is True
    assert guard["deep_mode_only"] is True
