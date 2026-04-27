"""Tests for PR16 deterministic process monitor / claimwise verification."""

from __future__ import annotations

from agent.process_monitor import (
    Claim,
    ProcessMonitorReport,
    assess_claims,
    build_process_monitor_context,
    build_process_monitor_metadata,
    extract_claims_from_response,
)


def test_extract_claims_blank_response_returns_empty_list():
    assert extract_claims_from_response("") == []
    assert extract_claims_from_response("   \n ") == []


def test_extract_claims_from_bullets_and_sentences_is_deterministic():
    response = """
    - I checked the repository status.
    - We will send the report tomorrow.
    This is based on the current branch.
    """

    claims = extract_claims_from_response(response, max_claims=5)

    assert [claim.text for claim in claims] == [
        "I checked the repository status.",
        "We will send the report tomorrow.",
        "This is based on the current branch.",
    ]


def test_extract_claims_classifies_action_status_policy_and_factual():
    response = """
    I checked the runtime status.
    I will send an email.
    Policy requires explicit confirmation.
    The branch is clean.
    """

    claims = extract_claims_from_response(response)

    assert [claim.kind for claim in claims] == ["status", "action", "policy", "factual"]


def test_assess_claims_flags_evidence_and_policy_gaps():
    claims = [
        Claim("The branch is clean.", kind="status"),
        Claim("I will send an email.", kind="action"),
        Claim("Policy requires confirmation.", kind="policy"),
    ]

    report = assess_claims(claims)

    assert isinstance(report, ProcessMonitorReport)
    assert report.enabled is True
    assert report.claim_count == 3
    assert report.evidence_gap_count == 2
    assert report.policy_gap_count == 2
    assert [a.rank for a in report.assessments] == [1, 2, 3]
    assert report.assessments[0].evidence_gap is True
    assert report.assessments[1].policy_gap is True
    assert report.assessments[2].policy_gap is True


def test_assess_claims_uses_evidence_and_policy_refs_to_reduce_gaps():
    claims = [
        Claim("The branch is clean.", kind="status"),
        Claim("I will send an email.", kind="action"),
    ]

    report = assess_claims(
        claims,
        evidence_refs=("git-status",),
        policy_refs=("policy:send-guard@1",),
    )

    assert report.supported_count == 2
    assert report.evidence_gap_count == 0
    assert report.policy_gap_count == 0
    assert all(a.supported for a in report.assessments)


def test_build_process_monitor_metadata_is_json_friendly():
    report = assess_claims(
        [
            Claim("The branch is clean.", kind="status"),
            Claim("I will send an email.", kind="action"),
        ],
        policy_refs=("policy:send-guard@1",),
    )

    metadata = build_process_monitor_metadata(report)

    assert metadata["process_monitor_enabled"] is True
    assert metadata["process_monitor_claim_count"] == 2
    assert metadata["process_monitor_evidence_gap_count"] == 2
    assert metadata["process_monitor_policy_gap_count"] == 0
    assert metadata["process_monitor_claim_kinds"] == ["status", "action"]
    assert "The branch is clean." in metadata["process_monitor_unsupported_claims"]
    assert metadata["process_monitor_policy_gap_claims"] == []


def test_build_process_monitor_context_includes_compact_counts():
    report = assess_claims([Claim("The branch is clean.", kind="status")])

    context = build_process_monitor_context(report)

    assert "claims=1" in context
    assert "evidence_gaps=1" in context
    assert "policy_gaps=0" in context
    assert "The branch is clean." in context
