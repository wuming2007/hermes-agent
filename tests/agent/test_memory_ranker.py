"""Tests for PR13 deterministic memory ranking primitives."""

from __future__ import annotations

from agent.memory_ranker import (
    MemoryCandidate,
    MemoryObjectMetadata,
    MemoryRankerConfig,
    build_ranked_memory_context,
    candidate_with_normalized_metadata,
    clamp_signal,
    memory_metadata_label,
    memory_metadata_to_dict,
    memory_tier_for_score,
    normalize_memory_metadata,
    rank_memory_candidates,
    score_memory_candidate,
)


def test_clamp_signal_handles_missing_invalid_and_bounds():
    assert clamp_signal(None) == 0.0
    assert clamp_signal("not-a-number") == 0.0
    assert clamp_signal(-0.25) == 0.0
    assert clamp_signal(1.25) == 1.0
    assert clamp_signal(0.4) == 0.4


def test_score_memory_candidate_rewards_positive_signals_and_penalizes_decay():
    low = MemoryCandidate(text="low", relevance=0.2, confidence=0.2)
    high = MemoryCandidate(
        text="high",
        relevance=0.9,
        recency=0.8,
        importance=0.7,
        reinforcement=0.6,
        confidence=0.9,
    )
    stale = MemoryCandidate(
        text="stale",
        relevance=0.9,
        recency=0.8,
        importance=0.7,
        reinforcement=0.6,
        confidence=0.9,
        decay_penalty=1.0,
    )

    assert score_memory_candidate(high) > score_memory_candidate(low)
    assert score_memory_candidate(stale) < score_memory_candidate(high)


def test_memory_tier_for_score_maps_to_hot_warm_cold_archive():
    assert memory_tier_for_score(0.85) == "hot"
    assert memory_tier_for_score(0.60) == "warm"
    assert memory_tier_for_score(0.35) == "cold"
    assert memory_tier_for_score(0.10) == "archive"


def test_rank_memory_candidates_is_deterministic_and_applies_max_items():
    config = MemoryRankerConfig(max_items=2)
    candidates = [
        MemoryCandidate(text="cold", provider="b", relevance=0.2),
        MemoryCandidate(text="hot", provider="a", relevance=0.9, confidence=0.9),
        MemoryCandidate(text="warm", provider="a", relevance=0.6, confidence=0.5),
    ]

    ranked = rank_memory_candidates(candidates, config=config)

    assert [item.candidate.text for item in ranked] == ["hot", "warm"]
    assert [item.rank for item in ranked] == [1, 2]
    assert ranked[0].score >= ranked[1].score


def test_rank_memory_candidates_disabled_preserves_input_order_but_bounds():
    config = MemoryRankerConfig(enabled=False, max_items=2)
    candidates = [
        MemoryCandidate(text="first", relevance=0.1),
        MemoryCandidate(text="second", relevance=0.9),
        MemoryCandidate(text="third", relevance=1.0),
    ]

    ranked = rank_memory_candidates(candidates, config=config)

    assert [item.candidate.text for item in ranked] == ["first", "second"]
    assert [item.rank for item in ranked] == [1, 2]


def test_build_ranked_memory_context_includes_metadata_and_respects_char_budget():
    config = MemoryRankerConfig(max_chars=140)
    ranked = rank_memory_candidates(
        [
            MemoryCandidate(
                text="important memory about Hermes cognition",
                provider="builtin",
                layer="semantic",
                source="MEMORY.md",
                relevance=1.0,
                confidence=0.9,
            ),
            MemoryCandidate(
                text="x" * 500,
                provider="external",
                relevance=0.8,
            ),
        ],
        config=MemoryRankerConfig(max_items=2),
    )

    context = build_ranked_memory_context(ranked, config=config)

    assert "[rank=1" in context
    assert "tier=" in context
    assert "provider=builtin" in context
    assert "source=MEMORY.md" in context
    assert "important memory" in context
    assert len(context) <= config.max_chars


# ---------------------------------------------------------------------------
# PR14 memory object metadata / source trace
# ---------------------------------------------------------------------------


def test_normalize_memory_metadata_defaults_to_unverified():
    metadata = normalize_memory_metadata(None)

    assert isinstance(metadata, MemoryObjectMetadata)
    assert metadata.source_trace == ()
    assert metadata.compression_level == 0
    assert metadata.confidence is None
    assert metadata.last_verified_at == ""
    assert metadata.superseded_by == ""
    assert metadata.reinforcement_count == 0
    assert metadata.status == "unverified"


def test_normalize_memory_metadata_from_dict_sanitizes_fields():
    metadata = normalize_memory_metadata(
        {
            "source_trace": ["MEMORY.md", 123, "USER.md"],
            "compression_level": "2",
            "confidence": 1.5,
            "last_verified_at": "2026-04-27T10:00:00+08:00",
            "superseded_by": "mem-2",
            "reinforcement_count": "3",
            "status": "active",
            "notes": {"kind": "preference"},
        }
    )

    assert metadata.source_trace == ("MEMORY.md", "123", "USER.md")
    assert metadata.compression_level == 2
    assert metadata.confidence == 1.0
    assert metadata.last_verified_at == "2026-04-27T10:00:00+08:00"
    assert metadata.superseded_by == "mem-2"
    assert metadata.reinforcement_count == 3
    assert metadata.status == "active"
    assert metadata.notes == {"kind": "preference"}


def test_normalize_memory_metadata_invalid_values_are_safe():
    metadata = normalize_memory_metadata(
        {
            "source_trace": "MEMORY.md > USER.md",
            "compression_level": -5,
            "confidence": "bad",
            "reinforcement_count": -9,
            "status": "mystery",
            "notes": "not-a-mapping",
        }
    )

    assert metadata.source_trace == ("MEMORY.md > USER.md",)
    assert metadata.compression_level == 0
    assert metadata.confidence is None
    assert metadata.reinforcement_count == 0
    assert metadata.status == "unverified"
    assert metadata.notes is None


def test_memory_metadata_to_dict_is_json_friendly():
    metadata = MemoryObjectMetadata(
        source_trace=("a", "b"),
        compression_level=1,
        confidence=0.75,
        last_verified_at="2026-04-27",
        superseded_by="newer",
        reinforcement_count=4,
        status="stale",
        notes={"why": "corrected"},
    )

    as_dict = memory_metadata_to_dict(metadata)

    assert as_dict == {
        "source_trace": ["a", "b"],
        "compression_level": 1,
        "confidence": 0.75,
        "last_verified_at": "2026-04-27",
        "superseded_by": "newer",
        "reinforcement_count": 4,
        "status": "stale",
        "notes": {"why": "corrected"},
    }


def test_candidate_with_normalized_metadata_does_not_mutate_original():
    candidate = MemoryCandidate(
        text="remember this",
        metadata={"source_trace": ["raw"], "confidence": 0.6, "status": "active"},
    )

    normalized = candidate_with_normalized_metadata(candidate)

    assert candidate.metadata == {"source_trace": ["raw"], "confidence": 0.6, "status": "active"}
    assert isinstance(normalized.metadata, MemoryObjectMetadata)
    assert normalized.metadata.source_trace == ("raw",)
    assert normalized.metadata.confidence == 0.6
    assert normalized.metadata.status == "active"


def test_memory_metadata_label_is_compact_and_auditable():
    metadata = MemoryObjectMetadata(
        source_trace=("MEMORY.md", "USER.md"),
        compression_level=1,
        confidence=0.8,
        last_verified_at="2026-04-27",
        superseded_by="mem-new",
        reinforcement_count=2,
        status="superseded",
    )

    label = memory_metadata_label(metadata)

    assert "status=superseded" in label
    assert "confidence=0.80" in label
    assert "verified=2026-04-27" in label
    assert "source_trace=MEMORY.md>USER.md" in label
    assert "superseded_by=mem-new" in label
    assert "compression=1" in label
    assert "reinforced=2" in label


def test_build_ranked_memory_context_includes_object_metadata_label():
    ranked = rank_memory_candidates(
        [
            MemoryCandidate(
                text="private gmail calendar rule",
                provider="builtin",
                object_id="mem-1",
                relevance=1.0,
                confidence=0.8,
                metadata={
                    "source_trace": ["USER.md"],
                    "confidence": 0.8,
                    "last_verified_at": "2026-04-27",
                    "status": "active",
                    "reinforcement_count": 2,
                },
            )
        ]
    )

    context = build_ranked_memory_context(ranked)

    assert "object_id=mem-1" in context
    assert "status=active" in context
    assert "confidence=0.80" in context
    assert "verified=2026-04-27" in context
    assert "source_trace=USER.md" in context
    assert "reinforced=2" in context
    assert "private gmail calendar rule" in context


def test_superseded_metadata_remains_visible_not_silently_dropped():
    ranked = rank_memory_candidates(
        [
            MemoryCandidate(
                text="old rule",
                relevance=1.0,
                metadata={"status": "superseded", "superseded_by": "new-rule"},
            )
        ]
    )

    context = build_ranked_memory_context(ranked)

    assert "old rule" in context
    assert "status=superseded" in context
    assert "superseded_by=new-rule" in context
