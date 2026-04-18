"""Tests for the PR3 consistency guard."""

from __future__ import annotations

from agent.cognitive_router import CognitiveRoute
from agent.consistency_guard import (
    VerificationResult,
    resolve_verification_plan,
    should_run_consistency_guard,
)


def _route(*, verification_plan="none", consistency_check=False, mode="standard"):
    return CognitiveRoute(
        mode=mode,
        retrieval_plan="principles_only",
        verification_plan=verification_plan,
        allow_cheap_model=False,
        consistency_check=consistency_check,
        routing_reasons=[],
    )


# ---------------------------------------------------------------------------
# Task 1: policy helpers
# ---------------------------------------------------------------------------


class TestResolveVerificationPlan:
    def test_none_route_returns_none(self):
        assert resolve_verification_plan(None) == "none"

    def test_route_with_plan_none(self):
        assert resolve_verification_plan(_route(verification_plan="none")) == "none"

    def test_route_with_plan_light(self):
        assert resolve_verification_plan(_route(verification_plan="light")) == "light"

    def test_route_with_plan_full(self):
        assert resolve_verification_plan(_route(verification_plan="full")) == "full"

    def test_unknown_plan_falls_back_to_none(self):
        # Forward-compat: a future router emitting an unknown string must not
        # crash existing deployments — they fall through to no-guard.
        bogus = _route(verification_plan="ultra")  # not in the Literal
        assert resolve_verification_plan(bogus) == "none"


class TestShouldRunConsistencyGuard:
    def test_none_route_returns_false(self):
        assert should_run_consistency_guard(None) is False

    def test_plan_none_returns_false(self):
        assert should_run_consistency_guard(_route(verification_plan="none")) is False

    def test_plan_light_returns_true(self):
        assert should_run_consistency_guard(_route(verification_plan="light")) is True

    def test_plan_full_returns_true(self):
        assert should_run_consistency_guard(_route(verification_plan="full")) is True

    def test_consistency_check_flag_alone_does_not_force_guard(self):
        # PR3 contract: verification_plan is the source of truth for whether
        # the guard runs. consistency_check is a separate orthogonal flag the
        # guard implementation may consult, but it must NOT force a guard run
        # on its own when verification_plan="none".
        r = _route(verification_plan="none", consistency_check=True)
        assert should_run_consistency_guard(r) is False


class TestVerificationResultDataclass:
    def test_defaults_make_a_passthrough_result(self):
        result = VerificationResult(
            applied=False,
            plan="none",
            original_response="hello",
            final_response="hello",
            changed=False,
        )
        assert result.notes == ()

    def test_is_frozen(self):
        import dataclasses
        result = VerificationResult(
            applied=False,
            plan="none",
            original_response="x",
            final_response="x",
            changed=False,
        )
        try:
            result.applied = True  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("VerificationResult must be frozen")
