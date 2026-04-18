"""Tests for the PR4 shared cognition config helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.cognition_config import (
    get_cognition_config,
    load_cognition_config_from_home,
)


# ---------------------------------------------------------------------------
# get_cognition_config
# ---------------------------------------------------------------------------


class TestGetCognitionConfig:
    def test_none_returns_empty_dict(self):
        assert get_cognition_config(None) == {}

    def test_non_dict_returns_empty_dict(self):
        assert get_cognition_config("nope") == {}
        assert get_cognition_config(42) == {}
        assert get_cognition_config([1, 2, 3]) == {}

    def test_missing_cognition_key_returns_empty_dict(self):
        assert get_cognition_config({"unrelated": True}) == {}

    def test_cognition_value_must_be_dict(self):
        # If someone wrote `cognition: true` in their YAML by mistake, we
        # should NOT crash and NOT pretend they configured anything.
        assert get_cognition_config({"cognition": True}) == {}
        assert get_cognition_config({"cognition": "enabled"}) == {}

    def test_passes_through_valid_block(self):
        cfg = {
            "cognition": {
                "enabled": True,
                "fast_mode": {"max_chars": 100},
                "deep_mode_triggers": {"historical_questions": True},
                "consistency_guard": {"enabled": True},
            }
        }
        result = get_cognition_config(cfg)
        assert result["enabled"] is True
        assert result["fast_mode"]["max_chars"] == 100
        assert result["deep_mode_triggers"]["historical_questions"] is True
        assert result["consistency_guard"]["enabled"] is True

    def test_returned_dict_is_a_copy(self):
        cfg = {"cognition": {"enabled": True, "fast_mode": {"max_chars": 100}}}
        result = get_cognition_config(cfg)
        result["enabled"] = False
        result["fast_mode"]["max_chars"] = 9999
        # Original config must not be mutated.
        assert cfg["cognition"]["enabled"] is True
        assert cfg["cognition"]["fast_mode"]["max_chars"] == 100

    def test_malformed_sub_block_becomes_empty_dict(self):
        # A list / string / int in a sub-block slot should not blow up the
        # router downstream — it normalizes to {} so .get() calls succeed.
        cfg = {
            "cognition": {
                "enabled": True,
                "fast_mode": "broken",
                "deep_mode_triggers": [1, 2, 3],
                "consistency_guard": 42,
            }
        }
        result = get_cognition_config(cfg)
        assert result["fast_mode"] == {}
        assert result["deep_mode_triggers"] == {}
        assert result["consistency_guard"] == {}
        assert result["enabled"] is True

    def test_partial_block_preserves_provided_keys(self):
        # Only enabled set; sub-blocks omitted entirely. Helper must NOT
        # invent default sub-blocks — callers that need defaults will read
        # via .get(..., default).
        cfg = {"cognition": {"enabled": True}}
        result = get_cognition_config(cfg)
        assert result == {"enabled": True}

    def test_disabled_block_normalized_to_disabled(self):
        cfg = {"cognition": {"enabled": False}}
        result = get_cognition_config(cfg)
        assert result["enabled"] is False


# ---------------------------------------------------------------------------
# load_cognition_config_from_home
# ---------------------------------------------------------------------------


class TestLoadCognitionConfigFromHome:
    def test_missing_config_file_returns_empty(self, tmp_path: Path):
        assert load_cognition_config_from_home(tmp_path) == {}

    def test_loads_block_from_yaml(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "cognition:\n  enabled: true\n  fast_mode:\n    max_chars: 100\n",
            encoding="utf-8",
        )
        result = load_cognition_config_from_home(tmp_path)
        assert result["enabled"] is True
        assert result["fast_mode"]["max_chars"] == 100

    def test_malformed_yaml_returns_empty_safely(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("::: not yaml at all", encoding="utf-8")
        assert load_cognition_config_from_home(tmp_path) == {}

    def test_yaml_without_cognition_returns_empty(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("model:\n  name: x\n", encoding="utf-8")
        assert load_cognition_config_from_home(tmp_path) == {}

    def test_disabled_block_loads_unchanged(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("cognition:\n  enabled: false\n", encoding="utf-8")
        result = load_cognition_config_from_home(tmp_path)
        assert result == {"enabled": False}
