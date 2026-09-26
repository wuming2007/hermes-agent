"""Per-turn cognitive routing for ``AIAgent`` (mixin; MRO-resolved from ``run_agent``).

Route resolve with uncertainty escalation (PR1/PR5) and the production verifier behind the
PR3 full consistency guard. Both are called by the turn phases (``agent.turn_context`` /
``agent.turn_finalizer``); with cognition disabled the route stays ``None``."""

import logging
from dataclasses import replace
from typing import Any, Dict, Iterable, List

logger = logging.getLogger("run_agent")  # origin module's name: log records / caplog filters unchanged

_PLAN_RANK = {"none": 0, "light": 1, "full": 2}
_BAND_RANK = {"low": 0, "medium": 1, "high": 2, "unknown": 3}


def merge_turn_guidance(pieces: Iterable[Any]) -> Dict[str, Any]:
    """One turn's ``turn_guidance`` from every ``pre_llm_call`` hook that gave some: the strictest
    answer check and the most cautious track-record band win."""
    merged: Dict[str, Any] = {}
    reasons: List[str] = []
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        floor = piece.get("verification_floor")
        if floor in _PLAN_RANK and _PLAN_RANK[floor] > _PLAN_RANK.get(merged.get("verification_floor"), -1):
            merged["verification_floor"] = floor
        band = piece.get("track_record_band")
        if band in _BAND_RANK and _BAND_RANK[band] < _BAND_RANK.get(merged.get("track_record_band"), len(_BAND_RANK)):
            merged["track_record_band"] = band
        reasons += [str(reason) for reason in piece.get("reasons") or () if reason]
    if reasons:
        merged["reasons"] = list(dict.fromkeys(reasons))
    return merged


def apply_verification_floor(route: Any, guidance: Dict[str, Any]) -> Any:
    """The route with its answer check raised to the guidance's floor. Only ever raised, and only the
    check: the mode, what is recalled and which model may answer stay as the router chose."""
    floor = (guidance or {}).get("verification_floor")
    if floor not in _PLAN_RANK or _PLAN_RANK[floor] <= _PLAN_RANK.get(route.verification_plan, 0):
        return route
    reasons = [f"track_record:{reason}" for reason in guidance.get("reasons") or ("floor",)]
    return replace(route, verification_plan=floor, routing_reasons=[*route.routing_reasons, *reasons])


class CognitionRoutingMixin:
    """``_resolve_current_cognitive_route`` + ``_default_consistency_verifier``."""

    # ── Per-turn cognitive routing (PR1) ────────────────────────────────

    def _resolve_current_cognitive_route(
        self,
        original_user_message: Any,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Compute and store the cognitive route for the current turn.

        No-op when cognition is disabled. Stores the resolved
        ``CognitiveRoute`` (or ``None``) on ``self._current_cognitive_route``
        and a flat ``dict`` snapshot on
        ``self._current_turn_cognition_metadata`` for downstream layers
        (telemetry, smart-model gating, layered retrieval) to consume.
        """
        from agent.cognitive_router import CognitiveRoute, resolve_cognitive_route
        from agent.uncertainty_policy import resolve_uncertainty_decision

        cognition_cfg = self._cognition_config or {}
        message_text = original_user_message if isinstance(original_user_message, str) else ""

        def _escalated_route(route: CognitiveRoute, target_mode: str | None) -> CognitiveRoute:
            if target_mode == "standard":
                return CognitiveRoute(
                    mode="standard",
                    retrieval_plan="principles_plus_semantic",
                    verification_plan="light",
                    allow_cheap_model=False,
                    consistency_check=False,
                    routing_reasons=list(route.routing_reasons) + ["uncertainty_escalated:standard"],
                    dialogue_mode=route.dialogue_mode,
                    answer_density=route.answer_density,
                    stance_reasons=list(route.stance_reasons),
                )
            if target_mode == "deep":
                consistency_cfg = cognition_cfg.get("consistency_guard") or {}
                consistency_enabled = consistency_cfg.get("enabled", True)
                return CognitiveRoute(
                    mode="deep",
                    retrieval_plan="principles_plus_semantic_plus_episodic",
                    verification_plan="full",
                    allow_cheap_model=False,
                    consistency_check=bool(consistency_enabled),
                    routing_reasons=list(route.routing_reasons) + ["uncertainty_escalated:deep"],
                    dialogue_mode=route.dialogue_mode,
                    answer_density=route.answer_density,
                    stance_reasons=list(route.stance_reasons),
                )
            return route

        try:
            route = resolve_cognitive_route(
                user_message=message_text,
                conversation_history=messages,
                routing_config=cognition_cfg,
                agent_state={
                    "platform": getattr(self, "platform", None),
                    "model": getattr(self, "model", None),
                    "provider": getattr(self, "provider", None),
                },
            )
        except Exception as e:  # router must never break the run loop
            logger.warning("cognitive_router failed; falling back to no route: %s", e)
            route = None

        uncertainty_decision = None
        original_mode = route.mode if route is not None else None
        if route is not None:
            try:
                uncertainty_decision = resolve_uncertainty_decision(
                    user_message=message_text,
                    cognition_route=route,
                    routing_config=cognition_cfg,
                    agent_state={
                        "platform": getattr(self, "platform", None),
                        "model": getattr(self, "model", None),
                        "provider": getattr(self, "provider", None),
                    },
                )
                if uncertainty_decision and uncertainty_decision.escalate_depth:
                    route = _escalated_route(route, uncertainty_decision.target_mode)
            except Exception as e:  # uncertainty policy must never break the run loop
                logger.warning("uncertainty_policy failed; keeping cognitive route: %s", e)
                uncertainty_decision = None

        # A plugin's record of how this kind of work went (``pre_llm_call`` → ``turn_guidance``) may
        # raise the answer check for this turn; the router's own reading of the message stands.
        guidance = getattr(self, "_turn_guidance", None) or {}
        if route is not None:
            route = apply_verification_floor(route, guidance)

        self._current_cognitive_route = route
        if route is None:
            self._current_turn_cognition_metadata = {"mode": "disabled"}
        else:
            # PR1 stores plan strings only; the layered-retrieval rewrite
            # (PR2) will consume retrieval_plan / verification_plan to drive
            # actual provider behavior. PR5 appends uncertainty-policy
            # metadata while preserving the final route snapshot as the
            # downstream source of truth.
            self._current_turn_cognition_metadata = {
                "mode": route.mode,
                "retrieval_plan": route.retrieval_plan,
                "verification_plan": route.verification_plan,
                "allow_cheap_model": route.allow_cheap_model,
                "consistency_check": route.consistency_check,
                "routing_reasons": list(route.routing_reasons),
                "dialogue_mode": route.dialogue_mode,
                "answer_density": route.answer_density,
                "stance_reasons": list(route.stance_reasons),
            }
            if "track_record_band" in guidance:
                self._current_turn_cognition_metadata["track_record_band"] = guidance["track_record_band"]
            if uncertainty_decision is not None:
                self._current_turn_cognition_metadata.update({
                    "original_mode": original_mode,
                    "uncertainty_confidence_band": uncertainty_decision.confidence_band,
                    "uncertainty_action": uncertainty_decision.action,
                    "uncertainty_reasons": list(uncertainty_decision.reasons),
                    "depth_escalated": uncertainty_decision.escalate_depth,
                    "target_mode": uncertainty_decision.target_mode,
                    "require_tool_evidence": uncertainty_decision.require_tool_evidence,
                    "seek_human": uncertainty_decision.seek_human,
                })

    def _default_consistency_verifier(self, prompt: str) -> str:
        """Production verifier wrapper for the PR3 full consistency guard.

        Calls the auxiliary text client (task ``"consistency_guard"`` with
        auto-detect fallback) and returns the raw model output. The guard
        layer parses the output as JSON; failures here are wrapped as
        exceptions so the guard's non-fatal handling kicks in and the turn
        falls back to the candidate response.
        """
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client(task="consistency_guard")
        if client is None or not model:
            raise RuntimeError("no auxiliary client available for consistency_guard")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
            timeout=30,
        )
        return (response.choices[0].message.content or "").strip()
