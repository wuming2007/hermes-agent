from agent.autonomy_telemetry import (
    AutonomySignal,
    AutonomyTelemetry,
    build_autonomy_context,
    build_autonomy_metadata,
    build_autonomy_telemetry_from_metadata,
    normalize_autonomy_signal,
    resolve_autonomy_telemetry,
)


def test_normalize_autonomy_signal_coerces_and_bounds_values():
    signal = normalize_autonomy_signal(
        {
            "requested_action": 123,
            "external_action": 1,
            "user_approval_present": "yes",
            "tool_evidence_present": 0,
            "policy_support_present": True,
            "process_evidence_gap_count": "9",
            "process_policy_gap_count": -3,
            "plasticity_promoted_count": "2",
            "plasticity_decayed_count": "4",
            "competence_band": "expert",
            "risk_level": "critical",
        }
    )

    assert signal == AutonomySignal(
        requested_action="123",
        external_action=True,
        user_approval_present=True,
        tool_evidence_present=False,
        policy_support_present=True,
        process_evidence_gap_count=9,
        process_policy_gap_count=0,
        plasticity_promoted_count=2,
        plasticity_decayed_count=4,
        competence_band="unknown",
        risk_level="low",
    )


def test_empty_signal_resolves_disabled_observe_telemetry():
    telemetry = resolve_autonomy_telemetry(AutonomySignal())

    assert telemetry == AutonomyTelemetry(
        enabled=False,
        autonomy_level="observe",
        competence_band="unknown",
        risk_level="low",
        external_action=False,
        approval_required=False,
        approval_present=False,
        evidence_required=False,
        evidence_present=False,
        policy_supported=False,
        intervention_reasons=(),
        self_model_notes=("no_autonomy_signal",),
    )


def test_internal_low_risk_signal_resolves_assist():
    telemetry = resolve_autonomy_telemetry(
        {"requested_action": "summarize notes", "competence_band": "medium", "risk_level": "low"}
    )

    assert telemetry.enabled is True
    assert telemetry.autonomy_level == "assist"
    assert telemetry.approval_required is False
    assert telemetry.evidence_required is False
    assert telemetry.self_model_notes == ("internal_low_risk",)


def test_external_action_without_approval_records_intervention_reasons():
    telemetry = resolve_autonomy_telemetry(
        {
            "requested_action": "send email",
            "external_action": True,
            "risk_level": "high",
            "process_evidence_gap_count": 1,
        }
    )

    assert telemetry.enabled is True
    assert telemetry.autonomy_level == "blocked_pending_evidence"
    assert telemetry.approval_required is True
    assert telemetry.approval_present is False
    assert telemetry.evidence_required is True
    assert telemetry.evidence_present is False
    assert telemetry.intervention_reasons == (
        "external_action_requires_approval",
        "evidence_required_missing",
    )


def test_external_action_with_approval_evidence_and_policy_support_can_act_with_approval():
    telemetry = resolve_autonomy_telemetry(
        {
            "requested_action": "publish update",
            "external_action": True,
            "user_approval_present": True,
            "tool_evidence_present": True,
            "policy_support_present": True,
            "risk_level": "medium",
            "competence_band": "high",
        }
    )

    assert telemetry.autonomy_level == "act_with_approval"
    assert telemetry.approval_required is True
    assert telemetry.approval_present is True
    assert telemetry.evidence_required is True
    assert telemetry.evidence_present is True
    assert telemetry.policy_supported is True
    assert telemetry.intervention_reasons == ()


def test_metadata_builder_emits_flat_autonomy_keys():
    telemetry = resolve_autonomy_telemetry(
        {"requested_action": "summarize", "competence_band": "low"}
    )

    assert build_autonomy_metadata(telemetry) == {
        "autonomy_enabled": True,
        "autonomy_level": "assist",
        "autonomy_competence_band": "low",
        "autonomy_risk_level": "low",
        "autonomy_external_action": False,
        "autonomy_approval_required": False,
        "autonomy_approval_present": False,
        "autonomy_evidence_required": False,
        "autonomy_evidence_present": False,
        "autonomy_policy_supported": False,
        "autonomy_intervention_reasons": [],
        "autonomy_self_model_notes": ["internal_low_risk"],
    }


def test_build_autonomy_telemetry_from_metadata_uses_existing_cognition_signals():
    telemetry = build_autonomy_telemetry_from_metadata(
        {
            "require_tool_evidence": True,
            "policy_memory_citations": ["policy:external@1"],
            "process_monitor_evidence_gap_count": 2,
            "process_monitor_policy_gap_count": 1,
            "plasticity_promoted_count": 1,
            "uncertainty_confidence_band": "low",
        }
    )

    assert telemetry.enabled is True
    assert telemetry.autonomy_level == "blocked_pending_evidence"
    assert telemetry.evidence_required is True
    assert telemetry.policy_supported is True
    assert telemetry.competence_band == "low"
    assert "process_gaps_present" in telemetry.self_model_notes


def test_context_builder_renders_compact_summary():
    telemetry = resolve_autonomy_telemetry(
        {"requested_action": "send email", "external_action": True, "risk_level": "high"}
    )

    context = build_autonomy_context(telemetry)

    assert "Autonomy Telemetry" in context
    assert "level=blocked_pending_evidence" in context
    assert "external_action=True" in context
