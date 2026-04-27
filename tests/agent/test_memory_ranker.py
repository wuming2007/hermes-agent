"""Tests for PR13 deterministic memory ranking primitives."""

from __future__ import annotations

from agent.memory_ranker import (
    MemoryCandidate,
    MemoryRankerConfig,
    build_ranked_memory_context,
    clamp_signal,
    memory_tier_for_score,
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
