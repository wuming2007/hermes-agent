"""Deterministic process monitor / claimwise verification primitives.

PR16 intentionally keeps this layer pure and observational: it extracts a small
bounded set of response claims, checks whether they have evidence/policy support,
and emits JSON-friendly metadata.  It never calls a model, never mutates the
response, and is safe to wrap fail-open from runtime wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Iterable, Sequence

_ALLOWED_KINDS = {"factual", "action", "status", "policy", "causal", "unknown"}
_ACTION_TERMS = (
    "will ",
    "i'll",
    "we'll",
    "send",
    "email",
    "publish",
    "post",
    "delete",
    "remove",
    "寄",
    "發送",
    "發布",
    "刪除",
)
_STATUS_TERMS = (
    "checked",
    "status",
    "done",
    "completed",
    "passed",
    "failed",
    "已完成",
    "狀態",
    "通過",
)
_POLICY_TERMS = ("policy", "requires", "must", "should", "confirmation", "confirm", "原則", "政策", "必須")
_CAUSAL_TERMS = ("because", "therefore", "root cause", "caused", "導致", "因為", "所以")


@dataclass(frozen=True)
class Claim:
    """A compact claim extracted from an assistant response."""

    text: str
    kind: str = "factual"
    evidence_refs: tuple[str, ...] = ()
    policy_refs: tuple[str, ...] = ()
    confidence: str = "unknown"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimAssessment:
    """Per-claim process-monitor assessment."""

    claim: Claim
    rank: int
    supported: bool
    evidence_gap: bool
    policy_gap: bool
    evidence_refs: tuple[str, ...] = ()
    policy_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessMonitorReport:
    """Aggregate process-monitor report."""

    enabled: bool
    claims: tuple[Claim, ...]
    assessments: tuple[ClaimAssessment, ...] = ()
    supported_count: int = 0
    evidence_gap_count: int = 0
    policy_gap_count: int = 0
    notes: tuple[str, ...] = ()

    @property
    def claim_count(self) -> int:
        return len(self.claims)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def _normalize_kind(kind: Any) -> str:
    text = str(kind or "").strip().lower()
    return text if text in _ALLOWED_KINDS else "unknown"


def _classify_claim(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in _POLICY_TERMS):
        return "policy"
    if any(term in lower for term in _ACTION_TERMS):
        return "action"
    if any(term in lower for term in _CAUSAL_TERMS):
        return "causal"
    if any(term in lower for term in _STATUS_TERMS):
        return "status"
    return "factual"


def _normalize_claim(value: Claim | str) -> Claim | None:
    if isinstance(value, Claim):
        text = value.text.strip()
        if not text:
            return None
        kind = _normalize_kind(value.kind)
        if kind == "unknown":
            kind = _classify_claim(text)
        return replace(
            value,
            text=text,
            kind=kind,
            evidence_refs=_as_str_tuple(value.evidence_refs),
            policy_refs=_as_str_tuple(value.policy_refs),
            notes=_as_str_tuple(value.notes),
        )
    text = str(value or "").strip()
    if not text:
        return None
    return Claim(text=text, kind=_classify_claim(text))


def _split_plain_text_claims(response: str) -> list[str]:
    claims: list[str] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        for part in parts:
            text = part.strip()
            if text:
                claims.append(text)
    return claims


def extract_claims_from_response(response: str, max_claims: int = 8) -> list[Claim]:
    """Extract a bounded deterministic claim list from final response text."""

    if not isinstance(response, str) or not response.strip():
        return []
    limit = max(0, int(max_claims or 0))
    if limit <= 0:
        return []
    claims: list[Claim] = []
    seen: set[str] = set()
    for text in _split_plain_text_claims(response):
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        claims.append(Claim(text=normalized, kind=_classify_claim(normalized)))
        if len(claims) >= limit:
            break
    return claims


def _policy_refs_from_metadata(policy_metadata: Any) -> tuple[str, ...]:
    if not isinstance(policy_metadata, dict):
        return ()
    return _as_str_tuple(
        policy_metadata.get("citations")
        or policy_metadata.get("policy_memory_citations")
        or policy_metadata.get("policy_refs")
    )


def assess_claims(
    claims: Sequence[Claim | str],
    *,
    evidence_refs: Sequence[str] | str = (),
    policy_refs: Sequence[str] | str = (),
    policy_metadata: dict[str, Any] | None = None,
    verification_notes: Sequence[str] | str = (),
) -> ProcessMonitorReport:
    """Assess whether claims have evidence/policy support.

    A v1 report is conservative: factual/status/action claims need evidence;
    action/policy claims also need policy support.  Existing per-claim refs,
    runtime evidence refs, policy citations, or verification notes can satisfy
    the relevant requirement.
    """

    normalized_claims = tuple(
        claim for claim in (_normalize_claim(item) for item in claims) if claim is not None
    )
    runtime_evidence = _as_str_tuple(evidence_refs) + _as_str_tuple(verification_notes)
    runtime_policy = _as_str_tuple(policy_refs) + _policy_refs_from_metadata(policy_metadata)
    assessments: list[ClaimAssessment] = []
    for index, claim in enumerate(normalized_claims, start=1):
        kind = _normalize_kind(claim.kind)
        needs_evidence = kind in {"factual", "status", "action", "causal", "unknown"}
        needs_policy = kind in {"action", "policy"}
        claim_evidence = _as_str_tuple(claim.evidence_refs) + runtime_evidence
        claim_policy = _as_str_tuple(claim.policy_refs) + runtime_policy
        evidence_gap = needs_evidence and not claim_evidence
        policy_gap = needs_policy and not claim_policy
        supported = not evidence_gap and not policy_gap
        notes: list[str] = []
        if evidence_gap:
            notes.append("missing_evidence")
        if policy_gap:
            notes.append("missing_policy")
        assessments.append(
            ClaimAssessment(
                claim=claim,
                rank=index,
                supported=supported,
                evidence_gap=evidence_gap,
                policy_gap=policy_gap,
                evidence_refs=claim_evidence,
                policy_refs=claim_policy,
                notes=tuple(notes),
            )
        )
    return ProcessMonitorReport(
        enabled=True,
        claims=normalized_claims,
        assessments=tuple(assessments),
        supported_count=sum(1 for item in assessments if item.supported),
        evidence_gap_count=sum(1 for item in assessments if item.evidence_gap),
        policy_gap_count=sum(1 for item in assessments if item.policy_gap),
    )


def build_process_monitor_metadata(report: ProcessMonitorReport | None) -> dict[str, Any]:
    """Convert a monitor report into flat JSON-friendly cognition metadata."""

    if not isinstance(report, ProcessMonitorReport):
        return {
            "process_monitor_enabled": False,
            "process_monitor_claim_count": 0,
            "process_monitor_supported_count": 0,
            "process_monitor_evidence_gap_count": 0,
            "process_monitor_policy_gap_count": 0,
            "process_monitor_claim_kinds": [],
            "process_monitor_unsupported_claims": [],
            "process_monitor_policy_gap_claims": [],
        }
    unsupported = [item.claim.text for item in report.assessments if not item.supported]
    policy_gaps = [item.claim.text for item in report.assessments if item.policy_gap]
    return {
        "process_monitor_enabled": bool(report.enabled),
        "process_monitor_claim_count": report.claim_count,
        "process_monitor_supported_count": int(report.supported_count),
        "process_monitor_evidence_gap_count": int(report.evidence_gap_count),
        "process_monitor_policy_gap_count": int(report.policy_gap_count),
        "process_monitor_claim_kinds": [claim.kind for claim in report.claims],
        "process_monitor_unsupported_claims": unsupported,
        "process_monitor_policy_gap_claims": policy_gaps,
    }


def build_process_monitor_context(report: ProcessMonitorReport | None) -> str:
    """Build a compact human/debug context string for a monitor report."""

    if not isinstance(report, ProcessMonitorReport):
        return "Process Monitor: disabled"
    lines = [
        "Process Monitor: "
        f"claims={report.claim_count} supported={report.supported_count} "
        f"evidence_gaps={report.evidence_gap_count} policy_gaps={report.policy_gap_count}"
    ]
    unsupported = [item.claim.text for item in report.assessments if not item.supported]
    policy_gaps = [item.claim.text for item in report.assessments if item.policy_gap]
    if unsupported:
        lines.append("unsupported: " + "; ".join(unsupported))
    if policy_gaps:
        lines.append("policy_gap_claims: " + "; ".join(policy_gaps))
    return "\n".join(lines)
