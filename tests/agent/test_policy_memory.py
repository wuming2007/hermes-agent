"""Tests for PR15 deterministic policy / constitution memory."""

from __future__ import annotations

from agent.memory_ranker import MemoryCandidate, MemoryObjectMetadata
from agent.policy_memory import (
    PolicyMemoryItem,
    build_policy_memory_context,
    build_policy_recall_metadata,
    normalize_policy_memory_item,
    policy_item_to_candidate,
    recall_policy_memories,
    score_policy_item,
)


def test_normalize_policy_memory_item_from_dict():
    item = normalize_policy_memory_item(
        {
            "policy_id": "external-email-guard",
            "title": "No email from cron",
            "text": "Scheduled reports must not send email unless explicitly requested.",
            "category": "external_action",
            "priority": 120,
            "scope": "user",
            "source_trace": ["SOUL.md", "USER.md"],
            "version": "2",
            "enabled": True,
            "tags": ["email", 123],
            "metadata": {"owner": "wuming"},
        }
    )

    assert isinstance(item, PolicyMemoryItem)
    assert item.policy_id == "external-email-guard"
    assert item.priority == 100
    assert item.source_trace == ("SOUL.md", "USER.md")
    assert item.tags == ("email", "123")
    assert item.metadata == {"owner": "wuming"}


def test_normalize_policy_memory_item_rejects_invalid_items():
    assert normalize_policy_memory_item(None) is None
    assert normalize_policy_memory_item({"policy_id": "x", "title": "", "text": "body"}) is None
    assert normalize_policy_memory_item({"policy_id": "x", "title": "title", "text": ""}) is None


def test_score_policy_item_detects_external_action_terms():
    item = PolicyMemoryItem(
        policy_id="send-guard",
        title="External action guard",
        text="Do not send email or publish public content without explicit confirmation.",
        category="external_action",
        priority=90,
        tags=("email", "send"),
    )

    score, terms = score_policy_item(item, "please send the morning report by email")

    assert score > 0
    assert "send" in terms
    assert "email" in terms


def test_recall_policy_memories_sorts_and_excludes_disabled_items():
    low = PolicyMemoryItem("low", "Low", "email rule", category="external_action", priority=10)
    high = PolicyMemoryItem("high", "High", "email send rule", category="external_action", priority=90)
    disabled = PolicyMemoryItem("off", "Off", "email", enabled=False, priority=100)

    results = recall_policy_memories("send email", [low, high, disabled], max_items=5)

    assert [r.item.policy_id for r in results] == ["high", "low"]
    assert all(r.item.policy_id != "off" for r in results)
    assert [r.rank for r in results] == [1, 2]
    assert results[0].citation == "policy:high@1"


def test_build_policy_memory_context_includes_citation_category_and_source_trace():
    item = PolicyMemoryItem(
        "p1",
        "Privacy",
        "Private data must not be exposed.",
        category="privacy",
        priority=80,
        source_trace=("SOUL.md",),
    )
    results = recall_policy_memories("private data", [item])

    context = build_policy_memory_context(results)

    assert "policy:p1@1" in context
    assert "category=privacy" in context
    assert "priority=80" in context
    assert "source_trace=SOUL.md" in context
    assert "Private data must not be exposed." in context


def test_policy_item_to_candidate_produces_principles_memory_candidate_with_metadata():
    item = PolicyMemoryItem(
        "p2",
        "Memory correction",
        "Corrections should supersede stale memory.",
        category="memory",
        priority=75,
        source_trace=("USER.md",),
        tags=("memory",),
    )

    candidate = policy_item_to_candidate(item, query="memory correction")

    assert isinstance(candidate, MemoryCandidate)
    assert candidate.provider == "policy"
    assert candidate.layer == "principles"
    assert candidate.source == "policy:p2"
    assert candidate.object_id == "p2"
    assert candidate.importance == 0.75
    assert isinstance(candidate.metadata, MemoryObjectMetadata)
    assert candidate.metadata.source_trace == ("USER.md",)
    assert candidate.metadata.status == "active"
    assert candidate.metadata.confidence == 1.0


def test_build_policy_recall_metadata_is_json_friendly():
    item = PolicyMemoryItem("p3", "Send", "Do not send email", category="external_action")
    results = recall_policy_memories("send email", [item])

    metadata = build_policy_recall_metadata(results)

    assert metadata == {
        "enabled": True,
        "count": 1,
        "policy_ids": ["p3"],
        "citations": ["policy:p3@1"],
        "categories": ["external_action"],
    }
