"""Tests for deterministic cognition turn trace snapshots."""

from copy import deepcopy

from agent.cognition_trace import build_cognition_turn_trace


def test_none_metadata_builds_disabled_trace():
    trace = build_cognition_turn_trace(None)

    assert trace["schema_version"] == 1
    assert trace["enabled"] is False
    assert trace["route"]["mode"] == "disabled"
    assert trace["uncertainty"]["present"] is False
    assert trace["interaction"] == {
        "dialogue_mode": "query",
        "answer_density": "standard",
        "stance_reasons": [],
    }
    assert trace["verification"]["ladder_enabled"] is False


def test_disabled_metadata_builds_disabled_trace():
    trace = build_cognition_turn_trace({"mode": "disabled"})

    assert trace["enabled"] is False
    assert trace["route"] == {
        "mode": "disabled",
        "original_mode": None,
        "retrieval_plan": None,
        "verification_plan": None,
        "allow_cheap_model": None,
        "consistency_check": None,
        "routing_reasons": [],
    }
    assert trace["uncertainty"]["present"] is False
    assert trace["verification"]["ladder_stages"] == []


def test_route_metadata_is_normalized():
    trace = build_cognition_turn_trace(
        {
            "mode": "standard",
            "retrieval_plan": "principles_plus_semantic",
            "verification_plan": "light",
            "allow_cheap_model": False,
            "consistency_check": True,
            "routing_reasons": ("semantic_needed", "standard_mode"),
        }
    )

    assert trace["enabled"] is True
    assert trace["route"] == {
        "mode": "standard",
        "original_mode": None,
        "retrieval_plan": "principles_plus_semantic",
        "verification_plan": "light",
        "allow_cheap_model": False,
        "consistency_check": True,
        "routing_reasons": ["semantic_needed", "standard_mode"],
    }


def test_interaction_metadata_is_grouped():
    trace = build_cognition_turn_trace(
        {
            "mode": "fast",
            "dialogue_mode": "status",
            "answer_density": "brief",
            "stance_reasons": ("status:目前", "route:fast"),
        }
    )

    assert trace["interaction"] == {
        "dialogue_mode": "status",
        "answer_density": "brief",
        "stance_reasons": ["status:目前", "route:fast"],
    }


def test_uncertainty_metadata_is_grouped_when_present():
    trace = build_cognition_turn_trace(
        {
            "mode": "standard",
            "original_mode": "fast",
            "uncertainty_confidence_band": "medium",
            "uncertainty_action": "escalate_depth",
            "uncertainty_reasons": ("historical_cue",),
            "depth_escalated": True,
            "target_mode": "standard",
            "require_tool_evidence": False,
            "seek_human": False,
        }
    )

    assert trace["uncertainty"] == {
        "present": True,
        "confidence_band": "medium",
        "action": "escalate_depth",
        "reasons": ["historical_cue"],
        "depth_escalated": True,
        "target_mode": "standard",
        "require_tool_evidence": False,
        "seek_human": False,
    }
    assert trace["route"]["original_mode"] == "fast"


def test_uncertainty_defaults_when_absent():
    trace = build_cognition_turn_trace({"mode": "fast"})

    assert trace["uncertainty"] == {
        "present": False,
        "confidence_band": None,
        "action": None,
        "reasons": [],
        "depth_escalated": False,
        "target_mode": None,
        "require_tool_evidence": False,
        "seek_human": False,
    }


def test_verification_metadata_is_grouped():
    trace = build_cognition_turn_trace(
        {
            "mode": "deep",
            "verification_ladder_enabled": True,
            "verification_ladder_source_plan": "full",
            "verification_ladder_stages": ("self_correction", "fast_monitor", "slow_verifier"),
            "verification_ladder_applied_stages": ("self_correction", "fast_monitor"),
            "verification_applied": True,
            "verification_changed": False,
            "verification_notes": ("ok",),
        }
    )

    assert trace["verification"] == {
        "ladder_enabled": True,
        "ladder_source_plan": "full",
        "ladder_stages": ["self_correction", "fast_monitor", "slow_verifier"],
        "ladder_applied_stages": ["self_correction", "fast_monitor"],
        "applied": True,
        "changed": False,
        "notes": ["ok"],
    }


def test_scalar_reason_values_are_coerced_to_string_lists():
    trace = build_cognition_turn_trace(
        {
            "mode": "standard",
            "routing_reasons": "single_reason",
            "uncertainty_reasons": "uncertain",
            "stance_reasons": "status:狀態",
            "verification_notes": "checked",
        }
    )

    assert trace["route"]["routing_reasons"] == ["single_reason"]
    assert trace["uncertainty"]["reasons"] == ["uncertain"]
    assert trace["interaction"]["stance_reasons"] == ["status:狀態"]
    assert trace["verification"]["notes"] == ["checked"]


def test_builder_does_not_mutate_metadata():
    metadata = {
        "mode": "standard",
        "routing_reasons": ["semantic_needed"],
        "verification_ladder_stages": ["self_correction", "fast_monitor"],
    }
    before = deepcopy(metadata)

    trace = build_cognition_turn_trace(metadata)

    assert metadata == before
    assert trace["route"]["routing_reasons"] == ["semantic_needed"]
