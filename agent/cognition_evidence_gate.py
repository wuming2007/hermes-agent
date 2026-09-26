"""Evidence stop gate: one bounded, read-only check before an answer states results the turn's
tools do not show, and an evidence note on whatever stays unverified.

Opt-in (``cognition.evidence_loop.enabled``). When the model stops with state claims that
``agent.cognition_evidence`` finds unsupported, ``agent.turn_stop_gates`` appends the candidate and
a flagged nudge (neither is stored nor shown, like the kanban gate) and the turn continues under a
dispatch-side policy that runs only read-only calls, at most ``max_extra_tool_calls``. The next
answer is final; claims it still cannot back get a short note appended before the answer is
stored, so what is stored is what the user sees.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional, Sequence

from agent.cognition_evidence import (
    ClaimCheck,
    EvidenceReport,
    build_evidence_report,
    claim_excerpt,
    command_changes_state,
)
from agent.tool_result_classification import NO_EFFECT_TOOL_NAMES

EVIDENCE_STOP_FLAG = "_evidence_stop_synthetic"
_DEFAULTS = {"enabled": False, "max_rounds": 1, "max_extra_tool_calls": 3}
_PROCESS_READ_ACTIONS = frozenset({"poll", "log", "wait", "list"})
_CJK = re.compile(r"[一-鿿]")

# reason code -> (for the model, for a Chinese answer)
REASON_TEXT = {
    "failed": ("the latest result failed", "最近一次結果是失敗"),
    "blocked": ("the action was blocked", "該動作被擋下"),
    "timeout": ("it timed out", "逾時，狀態未知"),
    "still_running": ("it is still running", "仍在執行，狀態未知"),
    "partial": ("it only partly succeeded", "只有部分成功"),
    "only_subagent_report": ("only a subagent's report says so", "只有子代理的回報"),
    "no_test_evidence": ("no test ran this turn", "本輪沒有跑測試"),
    "no_test_after_last_write": ("the tests ran before the last change", "測試是在最後一次修改前跑的"),
    "no_functional_check": ("nothing exercised it", "沒有實際執行驗證"),
    "no_functional_check_after_write": ("nothing exercised it after the last change", "最後一次修改後沒有實際執行驗證"),
    "no_check_evidence": ("nothing was checked", "沒有查證紀錄"),
    "no_check_after_last_write": ("nothing was checked after the last change", "最後一次修改後沒有查證"),
    "no_deploy_evidence": ("no deploy or restart ran this turn", "本輪沒有部署或重啟"),
    "no_write_evidence": ("no file was written this turn", "本輪沒有寫入檔案"),
    "no_action_evidence": ("no tool action ran this turn", "本輪沒有任何工具動作"),
}


def evidence_loop_config(agent: Any) -> dict:
    cognition = getattr(agent, "_cognition_config", None) or {}
    block = cognition.get("evidence_loop")
    return {**_DEFAULTS, **(block if isinstance(block, dict) else {})}


def _loop_enabled(agent: Any) -> bool:
    return bool(evidence_loop_config(agent)["enabled"]) and getattr(agent, "_current_cognitive_route", None) is not None


def actionable_claims(report: EvidenceReport) -> tuple[ClaimCheck, ...]:
    """Unsupported claims worth checking. A bare "done" in a turn that ran no tools is the answer
    itself (a translation, a draft), not a claim about the world."""
    return tuple(c for c in report.unsupported if report.evidence or c.claim_class != "completed")


def _read_only(tool_name: str, args: dict) -> bool:
    if tool_name in NO_EFFECT_TOOL_NAMES:
        return True
    if tool_name == "process_manage":
        return args.get("action") in _PROCESS_READ_ACTIONS
    if tool_name == "terminal":
        command = str(args.get("command") or "")
        # Conservative on purpose: any step that looks like a write or deploy blocks the command.
        return bool(command) and not args.get("background") and not command_changes_state(command)
    return False


class EvidenceRound:
    """Dispatch policy for the check round (``hermes_cli.plugins.set_thread_tool_policy``):
    read-only calls only, at most ``max_calls`` of them. Tool workers call it concurrently, and
    one call may be checked at more than one dispatch layer, so admissions count per call id."""

    def __init__(self, max_calls: int):
        self.max_calls = max(0, int(max_calls))
        self._admitted: list[str] = []
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return len(self._admitted)

    def __call__(self, tool_name: str, args: dict, tool_call_id: str = "") -> Optional[str]:
        if not _read_only(tool_name, args or {}):
            return (f"BLOCKED: evidence check — only read-only checks run now, so '{tool_name}' did not run. "
                    "Answer with what you could confirm and mark the rest unverified.")
        with self._lock:
            if tool_call_id and tool_call_id in self._admitted:
                return None
            if len(self._admitted) >= self.max_calls:
                return (f"BLOCKED: evidence check — the {self.max_calls} read-only checks are used up, so "
                        f"'{tool_name}' did not run. Give your final answer now.")
            self._admitted.append(tool_call_id or f"call-{len(self._admitted)}")
        return None


def evidence_nudge(agent: Any, final_response: Any, messages: Sequence[Any]) -> Optional[str]:
    """The check-round instruction for an answer with unsupported claims, or None."""
    config = evidence_loop_config(agent)
    if not _loop_enabled(agent) or getattr(agent, "_evidence_gate_rounds", 0) >= int(config["max_rounds"]):
        return None
    claims = actionable_claims(build_evidence_report(str(final_response or ""), messages))
    if not claims:
        return None
    listed = "\n".join(f'- {c.claim_class}: "{claim_excerpt(c, 120)}" ({REASON_TEXT[c.reason][0]})' for c in claims)
    return (
        "[Evidence check — automated, not from the user] Your answer states results that this turn's "
        f"tool output does not show:\n{listed}\n"
        f"Check them now with at most {int(config['max_extra_tool_calls'])} read-only tool calls (read files "
        "or logs, query status, poll a background job, run the tests); edits, deploys, restarts and "
        "messages are blocked. Then give your final answer in the user's language: keep what you "
        "confirmed and say plainly which of these you could not confirm or are still running."
    )


def begin_evidence_round(agent: Any) -> None:
    from hermes_cli.plugins import set_thread_tool_policy

    policy = EvidenceRound(evidence_loop_config(agent)["max_extra_tool_calls"])
    agent._evidence_round = policy
    set_thread_tool_policy(policy)


def end_evidence_round(agent: Any) -> None:
    """Lift the check-round policy once the model answers again (or the turn ends)."""
    policy = getattr(agent, "_evidence_round", None)
    if policy is None:
        return
    from hermes_cli.plugins import clear_thread_tool_whitelist

    agent._evidence_round = None
    agent._evidence_round_calls = getattr(agent, "_evidence_round_calls", 0) + policy.used
    clear_thread_tool_whitelist()


def evidence_note(agent: Any, answer: str, messages: Sequence[Any]) -> str:
    """The note appended to an answer whose state claims stay unverified ('' when none do)."""
    if not _loop_enabled(agent):
        return ""
    claims = actionable_claims(build_evidence_report(answer, messages))
    if not claims:
        return ""
    if _CJK.search(answer):
        listed = "\n".join(f"- {claim_excerpt(c, 120)}（{REASON_TEXT[c.reason][1]}）" for c in claims)
        return f"\n\n---\n證據狀態：以下說法本輪沒有工具結果佐證，請視為未證實：\n{listed}"
    listed = "\n".join(f"- {claim_excerpt(c, 120)} ({REASON_TEXT[c.reason][0]})" for c in claims)
    return f"\n\n---\nEvidence status: this turn's tool results do not show the following, so treat them as unverified:\n{listed}"


def evidence_loop_metadata(agent: Any) -> dict[str, Any]:
    return {
        "evidence_rounds_used": int(getattr(agent, "_evidence_gate_rounds", 0) or 0),
        "evidence_extra_tool_calls": int(getattr(agent, "_evidence_round_calls", 0) or 0),
    }
