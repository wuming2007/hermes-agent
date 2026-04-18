"""Tests for the PR3 consistency guard."""

from __future__ import annotations

from agent.cognitive_router import CognitiveRoute
from agent.consistency_guard import (
    VerificationResult,
    resolve_verification_plan,
    run_full_consistency_check,
    run_light_consistency_check,
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
        # PR4 contract (locked-in from PR3): verification_plan is the SINGLE
        # source of truth for whether the guard runs. consistency_check is a
        # non-execution hint that MUST NOT force a guard run on its own when
        # verification_plan="none".
        r = _route(verification_plan="none", consistency_check=True)
        assert should_run_consistency_guard(r) is False

    def test_full_plan_runs_guard_even_when_consistency_check_is_false(self):
        # PR4 contract symmetry: consistency_check=False MUST NOT suppress
        # a guard run when verification_plan asks for one. Future maintainers
        # who try to use consistency_check as a kill-switch will trip this
        # test and should re-read CognitiveRoute's contract docstring.
        r = _route(verification_plan="full", consistency_check=False)
        assert should_run_consistency_guard(r) is True

    def test_light_plan_runs_guard_even_when_consistency_check_is_false(self):
        r = _route(verification_plan="light", consistency_check=False)
        assert should_run_consistency_guard(r) is True

    def test_resolve_plan_does_not_consult_consistency_check(self):
        # Same plan string + opposite consistency_check values must produce
        # the same resolved plan — proves the resolver doesn't secretly
        # branch on consistency_check.
        a = _route(verification_plan="light", consistency_check=True)
        b = _route(verification_plan="light", consistency_check=False)
        assert resolve_verification_plan(a) == resolve_verification_plan(b)


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


# ---------------------------------------------------------------------------
# Task 2: light consistency check (rule-based)
# ---------------------------------------------------------------------------


class TestRunLightConsistencyCheck:
    def test_normal_response_passes_through_unchanged(self):
        result = run_light_consistency_check(
            candidate_response="Sure — Tokyo is currently 3pm JST.",
            user_message="what time is it in tokyo?",
        )
        assert result.applied is True
        assert result.plan == "light"
        assert result.changed is False
        assert result.original_response == result.final_response
        assert result.notes == ()

    def test_empty_response_is_flagged(self):
        result = run_light_consistency_check(
            candidate_response="",
            user_message="hello",
        )
        assert result.applied is True
        assert result.changed is False
        assert any("empty" in n.lower() for n in result.notes)

    def test_whitespace_only_response_is_flagged(self):
        result = run_light_consistency_check(
            candidate_response="   \n  \t",
            user_message="hello",
        )
        assert result.applied is True
        assert any("empty" in n.lower() for n in result.notes)

    def test_obvious_contradiction_pattern_is_flagged(self):
        # Same response claims both completion and not-yet-doing, which the
        # plan calls out explicitly as the kind of pattern light must catch.
        candidate = "I've completed the refactor. Actually I haven't started yet."
        result = run_light_consistency_check(
            candidate_response=candidate,
            user_message="did you finish?",
        )
        assert result.applied is True
        assert any("contradict" in n.lower() for n in result.notes)

    def test_chinese_contradiction_pattern_is_flagged(self):
        candidate = "已完成所有測試。其實我尚未做任何事。"
        result = run_light_consistency_check(
            candidate_response=candidate,
            user_message="完成了嗎？",
        )
        assert result.applied is True
        assert any("contradict" in n.lower() for n in result.notes)

    def test_light_does_not_rewrite_response_in_pr3(self):
        # Plan: light is rule-based and surfaces issues but does not
        # auto-repair in PR3 — that's reserved for full / future PRs.
        result = run_light_consistency_check(
            candidate_response="",
            user_message="hello",
        )
        assert result.changed is False
        assert result.final_response == ""

    def test_normal_long_response_does_not_trip_short_check(self):
        candidate = (
            "Here are the three main considerations for your migration: "
            "first, schema compatibility; second, downtime; third, rollback."
        )
        result = run_light_consistency_check(
            candidate_response=candidate, user_message="migration plan?"
        )
        assert result.notes == ()


# ---------------------------------------------------------------------------
# Task 3: full consistency check (mockable verifier call)
# ---------------------------------------------------------------------------


class TestRunFullConsistencyCheck:
    def test_verdict_ok_keeps_original_response(self):
        verifier = lambda prompt: {"verdict": "ok", "issues": []}
        result = run_full_consistency_check(
            candidate_response="Tokyo is 3pm.",
            user_message="time in tokyo?",
            verifier=verifier,
        )
        assert result.applied is True
        assert result.plan == "full"
        assert result.changed is False
        assert result.final_response == "Tokyo is 3pm."

    def test_verdict_revise_overrides_response(self):
        verifier = lambda prompt: {
            "verdict": "revise",
            "issues": ["wrong timezone"],
            "revised_response": "Tokyo is 3pm JST.",
        }
        result = run_full_consistency_check(
            candidate_response="Tokyo is 3pm.",
            user_message="time in tokyo?",
            verifier=verifier,
        )
        assert result.applied is True
        assert result.changed is True
        assert result.final_response == "Tokyo is 3pm JST."
        assert any("wrong timezone" in n for n in result.notes)

    def test_revise_without_revised_response_keeps_original(self):
        # If the verifier asks to revise but doesn't provide replacement
        # text, we cannot safely rewrite — keep candidate, surface a note.
        verifier = lambda prompt: {"verdict": "revise", "issues": ["incomplete"]}
        result = run_full_consistency_check(
            candidate_response="Half answer.",
            user_message="full answer?",
            verifier=verifier,
        )
        assert result.changed is False
        assert result.final_response == "Half answer."
        assert any("missing_revised_response" in n for n in result.notes)

    def test_verifier_exception_falls_back_to_original(self):
        def boom(prompt):
            raise RuntimeError("verifier upstream timeout")

        result = run_full_consistency_check(
            candidate_response="Original answer",
            user_message="?",
            verifier=boom,
        )
        # Non-fatal: turn must complete with the candidate response.
        assert result.applied is True  # still ran (and failed gracefully)
        assert result.changed is False
        assert result.final_response == "Original answer"
        assert any("verifier_error" in n for n in result.notes)

    def test_verifier_returns_garbage_falls_back(self):
        verifier = lambda prompt: "not even a dict"
        result = run_full_consistency_check(
            candidate_response="Original",
            user_message="?",
            verifier=verifier,
        )
        assert result.changed is False
        assert result.final_response == "Original"
        assert any("verifier_parse" in n for n in result.notes)

    def test_empty_candidate_short_circuits_without_verifier_call(self):
        called = []

        def verifier(prompt):
            called.append(prompt)
            return {"verdict": "ok"}

        result = run_full_consistency_check(
            candidate_response="",
            user_message="?",
            verifier=verifier,
        )
        # No point asking a verifier about an empty candidate.
        assert called == []
        assert result.changed is False
        assert any("empty_response" in n for n in result.notes)

    def test_revise_with_blank_revised_response_is_treated_as_missing(self):
        verifier = lambda prompt: {
            "verdict": "revise",
            "revised_response": "   \n  ",
            "issues": ["x"],
        }
        result = run_full_consistency_check(
            candidate_response="Original",
            user_message="?",
            verifier=verifier,
        )
        assert result.changed is False
        assert result.final_response == "Original"
        assert any("missing_revised_response" in n for n in result.notes)
