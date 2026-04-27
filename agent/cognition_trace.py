"""Deterministic cognition turn trace snapshots.

PR7 keeps the existing flat cognition metadata intact, but also exposes a
stable nested trace shape for telemetry, trajectory, and future adaptation
layers.  This module is intentionally pure: no logging, no IO, and no runtime
state mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def build_cognition_turn_trace(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a stable PR7 cognition trace from flat turn metadata.

    The input is treated as read-only. Unknown metadata keys are ignored so the
    flat schema can keep evolving without breaking trace consumers.
    """

    meta: Mapping[str, Any]
    if isinstance(metadata, Mapping):
        meta = metadata
    else:
        meta = {}

    mode = _as_optional_str(meta.get("mode")) or "disabled"
    enabled = mode != "disabled"

    uncertainty_present = any(
        key in meta
        for key in (
            "uncertainty_confidence_band",
            "uncertainty_action",
            "uncertainty_reasons",
            "depth_escalated",
            "target_mode",
            "require_tool_evidence",
            "seek_human",
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": enabled,
        "route": {
            "mode": mode,
            "original_mode": _as_optional_str(meta.get("original_mode")),
            "retrieval_plan": _as_optional_str(meta.get("retrieval_plan")),
            "verification_plan": _as_optional_str(meta.get("verification_plan")),
            "allow_cheap_model": _as_optional_bool(meta.get("allow_cheap_model")),
            "consistency_check": _as_optional_bool(meta.get("consistency_check")),
            "routing_reasons": _as_str_list(meta.get("routing_reasons")),
        },
        "interaction": {
            "dialogue_mode": _as_optional_str(meta.get("dialogue_mode")) or "query",
            "answer_density": _as_optional_str(meta.get("answer_density")) or "standard",
            "stance_reasons": _as_str_list(meta.get("stance_reasons")),
        },
        "uncertainty": {
            "present": uncertainty_present,
            "confidence_band": _as_optional_str(meta.get("uncertainty_confidence_band")),
            "action": _as_optional_str(meta.get("uncertainty_action")),
            "reasons": _as_str_list(meta.get("uncertainty_reasons")),
            "depth_escalated": _as_bool(meta.get("depth_escalated"), False),
            "target_mode": _as_optional_str(meta.get("target_mode")),
            "require_tool_evidence": _as_bool(meta.get("require_tool_evidence"), False),
            "seek_human": _as_bool(meta.get("seek_human"), False),
        },
        "verification": {
            "ladder_enabled": _as_bool(meta.get("verification_ladder_enabled"), False),
            "ladder_source_plan": _as_optional_str(
                meta.get("verification_ladder_source_plan")
            ),
            "ladder_stages": _as_str_list(meta.get("verification_ladder_stages")),
            "ladder_applied_stages": _as_str_list(
                meta.get("verification_ladder_applied_stages")
            ),
            "applied": _as_optional_bool(meta.get("verification_applied")),
            "changed": _as_optional_bool(meta.get("verification_changed")),
            "notes": _as_str_list(meta.get("verification_notes")),
        },
        "policy": {
            "enabled": _as_bool(meta.get("policy_memory_enabled"), False),
            "count": int(meta.get("policy_memory_count") or 0),
            "policy_ids": _as_str_list(meta.get("policy_memory_ids")),
            "citations": _as_str_list(meta.get("policy_memory_citations")),
            "categories": _as_str_list(meta.get("policy_memory_categories")),
        },
    }
