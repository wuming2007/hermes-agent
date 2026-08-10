"""Deterministic autonomy / self-model telemetry for PR18.

V1 is observation only: it turns bounded cognition metadata into JSON-friendly
telemetry. It does not grant permissions, execute tools, or persist memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_COMPETENCE_BANDS = {"unknown", "low", "medium", "high"}
_RISK_LEVELS = {"low", "medium", "high"}
_AUTONOMY_LEVELS = {"observe", "assist", "act_with_approval", "blocked_pending_evidence"}
_TRUE_STRINGS = {"1", "true", "yes", "y", "on", "approved", "present"}


@dataclass(frozen=True)
class AutonomySignal:
    requested_action: str = ""
    external_action: bool = False
    user_approval_present: bool = False
    tool_evidence_present: bool = False
    policy_support_present: bool = False
    process_evidence_gap_count: int = 0
    process_policy_gap_count: int = 0
    plasticity_promoted_count: int = 0
    plasticity_decayed_count: int = 0
    competence_band: str = "unknown"
    risk_level: str = "low"


@dataclass(frozen=True)
class AutonomyTelemetry:
    enabled: bool = False
    autonomy_level: str = "observe"
    competence_band: str = "unknown"
    risk_level: str = "low"
    external_action: bool = False
    approval_required: bool = False
    approval_present: bool = False
    evidence_required: bool = False
    evidence_present: bool = False
    policy_supported: bool = False
    intervention_reasons: tuple[str, ...] = ()
    self_model_notes: tuple[str, ...] = ()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _as_int(value: Any, *, max_value: int = 999) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return max(0, min(max_value, number))


def _bounded_str(value: Any, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text if text in allowed else default


def _has_signal(signal: AutonomySignal) -> bool:
    return any(
        (
            signal.requested_action,
            signal.external_action,
            signal.user_approval_present,
            signal.tool_evidence_present,
            signal.policy_support_present,
            signal.process_evidence_gap_count,
            signal.process_policy_gap_count,
            signal.plasticity_promoted_count,
            signal.plasticity_decayed_count,
            signal.competence_band != "unknown",
            signal.risk_level != "low",
        )
    )


def normalize_autonomy_signal(value: Any) -> AutonomySignal:
    """Normalize arbitrary metadata into a bounded autonomy signal."""

    if isinstance(value, AutonomySignal):
        return value
    if not isinstance(value, Mapping):
        return AutonomySignal()
    return AutonomySignal(
        requested_action=str(value.get("requested_action") or ""),
        external_action=_as_bool(value.get("external_action")),
        user_approval_present=_as_bool(value.get("user_approval_present")),
        tool_evidence_present=_as_bool(value.get("tool_evidence_present")),
        policy_support_present=_as_bool(value.get("policy_support_present")),
        process_evidence_gap_count=_as_int(value.get("process_evidence_gap_count")),
        process_policy_gap_count=_as_int(value.get("process_policy_gap_count")),
        plasticity_promoted_count=_as_int(value.get("plasticity_promoted_count")),
        plasticity_decayed_count=_as_int(value.get("plasticity_decayed_count")),
        competence_band=_bounded_str(value.get("competence_band"), _COMPETENCE_BANDS, "unknown"),
        risk_level=_bounded_str(value.get("risk_level"), _RISK_LEVELS, "low"),
    )


def resolve_autonomy_telemetry(
    signal: AutonomySignal | Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> AutonomyTelemetry:
    """Resolve autonomy telemetry without changing runtime behavior."""

    sig = normalize_autonomy_signal(signal)
    meta = metadata if isinstance(metadata, Mapping) else {}
    if not _has_signal(sig) and not meta:
        return AutonomyTelemetry(self_model_notes=("no_autonomy_signal",))

    external_action = sig.external_action
    approval_required = external_action
    approval_present = sig.user_approval_present
    policy_supported = sig.policy_support_present
    evidence_present = sig.tool_evidence_present
    evidence_required = (
        sig.risk_level in {"medium", "high"}
        or sig.process_evidence_gap_count > 0
        or sig.process_policy_gap_count > 0
        or _as_bool(meta.get("require_tool_evidence"))
    )

    intervention: list[str] = []
    notes: list[str] = []
    if external_action and not approval_present:
        intervention.append("external_action_requires_approval")
    if evidence_required and not evidence_present:
        intervention.append("evidence_required_missing")
    if sig.process_evidence_gap_count or sig.process_policy_gap_count:
        notes.append("process_gaps_present")
    if policy_supported:
        notes.append("policy_supported")
    if sig.plasticity_promoted_count:
        notes.append("plasticity_promoted")
    if sig.plasticity_decayed_count:
        notes.append("plasticity_decayed")

    if intervention:
        level = "blocked_pending_evidence"
    elif external_action:
        level = "act_with_approval"
    else:
        level = "assist"
        if not notes:
            notes.append("internal_low_risk")

    return AutonomyTelemetry(
        enabled=True,
        autonomy_level=level,
        competence_band=sig.competence_band,
        risk_level=sig.risk_level,
        external_action=external_action,
        approval_required=approval_required,
        approval_present=approval_present,
        evidence_required=evidence_required,
        evidence_present=evidence_present,
        policy_supported=policy_supported,
        intervention_reasons=tuple(intervention),
        self_model_notes=tuple(notes),
    )


def build_autonomy_metadata(telemetry: AutonomyTelemetry | None) -> dict[str, Any]:
    """Build flat JSON-friendly autonomy metadata keys."""

    t = telemetry if isinstance(telemetry, AutonomyTelemetry) else AutonomyTelemetry()
    return {
        "autonomy_enabled": bool(t.enabled),
        "autonomy_level": t.autonomy_level if t.autonomy_level in _AUTONOMY_LEVELS else "observe",
        "autonomy_competence_band": t.competence_band if t.competence_band in _COMPETENCE_BANDS else "unknown",
        "autonomy_risk_level": t.risk_level if t.risk_level in _RISK_LEVELS else "low",
        "autonomy_external_action": bool(t.external_action),
        "autonomy_approval_required": bool(t.approval_required),
        "autonomy_approval_present": bool(t.approval_present),
        "autonomy_evidence_required": bool(t.evidence_required),
        "autonomy_evidence_present": bool(t.evidence_present),
        "autonomy_policy_supported": bool(t.policy_supported),
        "autonomy_intervention_reasons": [str(item) for item in t.intervention_reasons],
        "autonomy_self_model_notes": [str(item) for item in t.self_model_notes],
    }


def build_autonomy_context(telemetry: AutonomyTelemetry | None) -> str:
    """Render a compact autonomy telemetry summary."""

    if not isinstance(telemetry, AutonomyTelemetry) or not telemetry.enabled:
        return ""
    reasons = ",".join(telemetry.intervention_reasons) if telemetry.intervention_reasons else "none"
    notes = ",".join(telemetry.self_model_notes) if telemetry.self_model_notes else "none"
    return (
        "Autonomy Telemetry\n"
        f"level={telemetry.autonomy_level} competence={telemetry.competence_band} "
        f"risk={telemetry.risk_level} external_action={telemetry.external_action}\n"
        f"approval_required={telemetry.approval_required} approval_present={telemetry.approval_present} "
        f"evidence_required={telemetry.evidence_required} evidence_present={telemetry.evidence_present} "
        f"policy_supported={telemetry.policy_supported}\n"
        f"intervention_reasons={reasons} self_model_notes={notes}"
    )


def build_autonomy_telemetry_from_metadata(metadata: Mapping[str, Any] | None) -> AutonomyTelemetry:
    """Derive telemetry from existing cognition metadata."""

    meta = metadata if isinstance(metadata, Mapping) else {}
    if not meta:
        return resolve_autonomy_telemetry(AutonomySignal())

    explicit_external = _as_bool(meta.get("autonomy_external_action_hint"))
    action_text = str(meta.get("autonomy_requested_action") or "")
    if not explicit_external and action_text:
        lowered = action_text.lower()
        explicit_external = any(
            token in lowered
            for token in ("send", "email", "publish", "post", "tweet", "delete", "寄", "發送", "發布", "刪")
        )

    signal = AutonomySignal(
        requested_action=action_text,
        external_action=explicit_external,
        user_approval_present=_as_bool(meta.get("autonomy_user_approval_present")),
        tool_evidence_present=_as_bool(meta.get("tool_evidence_present"))
        or bool(meta.get("verification_notes")),
        policy_support_present=bool(meta.get("policy_memory_citations")),
        process_evidence_gap_count=_as_int(meta.get("process_monitor_evidence_gap_count")),
        process_policy_gap_count=_as_int(meta.get("process_monitor_policy_gap_count")),
        plasticity_promoted_count=_as_int(meta.get("plasticity_promoted_count")),
        plasticity_decayed_count=_as_int(meta.get("plasticity_decayed_count")),
        competence_band=_bounded_str(meta.get("uncertainty_confidence_band"), _COMPETENCE_BANDS, "unknown"),
        risk_level=_bounded_str(meta.get("autonomy_risk_level_hint"), _RISK_LEVELS, "low"),
    )
    return resolve_autonomy_telemetry(signal, metadata=meta)
