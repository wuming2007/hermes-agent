"""Tests for the PR2 retrieval policy mapping."""

from __future__ import annotations

import pytest

from agent.cognitive_router import CognitiveRoute
from agent.retrieval_policy import RetrievalPolicy, resolve_retrieval_policy


def _route(plan, *, mode="fast"):
    return CognitiveRoute(
        mode=mode,
        retrieval_plan=plan,
        verification_plan="none",
        allow_cheap_model=True,
        consistency_check=False,
        routing_reasons=[],
    )


def test_resolve_returns_none_when_route_is_none():
    assert resolve_retrieval_policy(None) is None


def test_principles_only_maps_to_principles_layer():
    policy = resolve_retrieval_policy(_route("principles_only", mode="fast"))
    assert isinstance(policy, RetrievalPolicy)
    assert policy.layers == ("principles",)


def test_principles_plus_semantic_maps_to_two_layers():
    policy = resolve_retrieval_policy(
        _route("principles_plus_semantic", mode="standard")
    )
    assert policy is not None
    assert policy.layers == ("principles", "semantic")


def test_principles_plus_semantic_plus_episodic_maps_to_three_layers():
    policy = resolve_retrieval_policy(
        _route("principles_plus_semantic_plus_episodic", mode="deep")
    )
    assert policy is not None
    assert policy.layers == ("principles", "semantic", "episodic")


def test_none_retrieval_plan_returns_no_policy():
    # The router currently never emits "none", but the type allows it; in that
    # case we must NOT return an empty-layers policy that would skip prefetch
    # entirely — fall through to legacy prefetch_all.
    policy = resolve_retrieval_policy(_route("none", mode="fast"))
    assert policy is None


def test_unknown_plan_returns_none_for_safe_legacy_fallback():
    policy = resolve_retrieval_policy(_route("future_unknown_plan_v9", mode="deep"))
    assert policy is None


def test_policy_carries_plan_string_for_diagnostics():
    policy = resolve_retrieval_policy(_route("principles_only"))
    assert policy is not None
    assert policy.plan == "principles_only"


def test_policy_layers_is_immutable_tuple():
    policy = resolve_retrieval_policy(_route("principles_plus_semantic"))
    assert policy is not None
    assert isinstance(policy.layers, tuple)
    with pytest.raises((TypeError, AttributeError)):
        # tuple has no append; verifies we don't accidentally hand out a list.
        policy.layers.append("episodic")  # type: ignore[attr-defined]
