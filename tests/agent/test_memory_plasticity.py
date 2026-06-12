"""Tests for PR17 deterministic memory plasticity primitives."""

from agent.memory_plasticity import (
    PlasticityConfig,
    PlasticityDecision,
    PlasticitySignal,
    apply_plasticity_to_candidate,
    build_plasticity_context,
    build_plasticity_metadata,
    normalize_plasticity_signal,
    resolve_plasticity_decision,
)
from agent.memory_ranker import MemoryCandidate, MemoryObjectMetadata


def test_normalize_plasticity_signal_clamps_and_coerces_values():
    signal = normalize_plasticity_signal(
        {
            "success_count": "3",
            "correction_count": -2,
            "verification_count": 4,
            "days_since_verified": "999",
            "explicit_decay": 2,
            "superseded_by": "new-object",
            "confidence_delta": "0.8",
        }
    )

    assert signal == PlasticitySignal(
        success_count=3,
        correction_count=0,
        verification_count=4,
        days_since_verified=365,
        explicit_decay=1.0,
        superseded_by="new-object",
        confidence_delta=0.5,
    )


def test_success_and_verification_promote_memory():
    decision = resolve_plasticity_decision(
        PlasticitySignal(success_count=3, verification_count=1),
        PlasticityConfig(promotion_threshold=0.4),
    )

    assert decision.action == "promote"
    assert decision.reinforcement_delta > 0
    assert decision.confidence_delta > 0
    assert decision.decay_delta < 0
    assert "success" in decision.reasons


def test_corrections_and_staleness_decay_memory():
    decision = resolve_plasticity_decision(
        PlasticitySignal(correction_count=2, days_since_verified=120)
    )

    assert decision.action == "decay"
    assert decision.decay_delta > 0
    assert decision.status == "stale"
    assert "correction" in decision.reasons


def test_superseded_signal_wins_over_other_actions():
    decision = resolve_plasticity_decision(
        PlasticitySignal(success_count=5, superseded_by="new-id")
    )

    assert decision.action == "supersede"
    assert decision.status == "superseded"
    assert decision.superseded_by == "new-id"


def test_apply_plasticity_to_candidate_updates_signals_and_metadata_notes():
    candidate = MemoryCandidate(
        text="old but useful memory",
        provider="p",
        relevance=0.5,
        reinforcement=0.2,
        confidence=0.4,
        decay_penalty=0.3,
        metadata=MemoryObjectMetadata(
            source_trace=("MEMORY.md",),
            confidence=0.4,
            reinforcement_count=2,
            status="active",
            notes={"plasticity": {"success_count": 3, "verification_count": 1}},
        ),
    )

    updated, decision = apply_plasticity_to_candidate(candidate)

    assert decision.action == "promote"
    assert updated.reinforcement > candidate.reinforcement
    assert updated.confidence > candidate.confidence
    assert updated.decay_penalty < candidate.decay_penalty
    assert isinstance(updated.metadata, MemoryObjectMetadata)
    assert updated.metadata.reinforcement_count > candidate.metadata.reinforcement_count
    assert updated.metadata.status == "active"
    assert updated.metadata.notes["plasticity_action"] == "promote"
    assert "success" in updated.metadata.notes["plasticity_reasons"]


def test_apply_supersede_sets_metadata_status_without_deleting_candidate():
    candidate = MemoryCandidate(
        text="superseded memory",
        provider="p",
        metadata={"notes": {"plasticity": {"superseded_by": "new-memory"}}},
    )

    updated, decision = apply_plasticity_to_candidate(candidate)

    assert updated.text == "superseded memory"
    assert decision.action == "supersede"
    assert isinstance(updated.metadata, MemoryObjectMetadata)
    assert updated.metadata.status == "superseded"
    assert updated.metadata.superseded_by == "new-memory"


def test_build_plasticity_metadata_and_context_are_json_friendly():
    decisions = [
        PlasticityDecision(action="promote", reinforcement_delta=0.2, reasons=("success",)),
        PlasticityDecision(action="decay", decay_delta=0.3, status="stale", reasons=("correction",)),
        PlasticityDecision(action="supersede", status="superseded", superseded_by="new", reasons=("superseded",)),
    ]

    metadata = build_plasticity_metadata(decisions)
    context = build_plasticity_context(decisions)

    assert metadata == {
        "plasticity_enabled": True,
        "plasticity_decision_count": 3,
        "plasticity_actions": ["promote", "decay", "supersede"],
        "plasticity_promoted_count": 1,
        "plasticity_decayed_count": 1,
        "plasticity_superseded_count": 1,
    }
    assert "Memory Plasticity" in context
    assert "promote" in context
    assert "supersede" in context
