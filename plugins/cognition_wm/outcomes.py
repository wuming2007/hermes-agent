"""How this turn's claimed results turned out, and which lessons were followed (cognition v2, phase 2).

Pure: the turn's evidence report goes in, outcome drafts come out. Our own claims count by what
stands behind them: a backed claim is a success; a failed, blocked or partial step is exactly that;
anything short of a shown result is unknown. A subagent's or coding agent's report counts for
whoever reported it, as reported, until our own check confirms or refutes it. An earlier session's
record is not this turn's result and counts for nothing. The same check seen again (a retry, a
re-run with the same output) rests on the same key, so it counts once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from agent.cognition_claims import state_claims
from agent.cognition_evidence import EvidenceItem, EvidenceReport

# our own claim's reason for lacking support -> its result; anything not listed is unknown
_SELF_RESULTS = {"failed": "failure", "blocked": "blocked", "partial": "partial"}
# a delegated run that did not end well -> its result
_REPORT_STATUS = {"error": "failure", "timeout": "unknown", "running": "unknown", "blocked": "blocked",
                  "partial": "partial"}
_WORST_FIRST = ("failure", "partial", "blocked", "unknown", "reported", "success")
_SHELL_TOOLS = frozenset({"terminal", "process_manage", "process_notice"})
_AGENT_NAME = re.compile(r"(?<![\w.-])(claude|codex|opencode|aider|gemini|cursor-agent|hermes)(?:-code|-cli|-chat|-ai)?"
                         r"(?![\w.-])")
_MODEL_FIELD = re.compile(r'"model"\s*:\s*"([^"]{1,80})"')
_NUMBERS = re.compile(r"\d+(?:\.\d+)?")
_BACKTICKED = re.compile(r"`([^`]{2,80})`")
_COMMAND_RUN = re.compile(r"[A-Za-z0-9][\w./:=+-]*(?:[ \t]+[A-Za-z0-9-][\w./:=+-]*)+")
_PROSE = frozenset("""
the a an of to in on for with instead use using always never before after and or but is are be should please then
when it that this here there not don't do does
""".split())


@dataclass(frozen=True)
class OutcomeDraft:
    claim_class: str
    actor_kind: str  # self | subagent | external_agent
    actor_id: str
    result: str
    reason: str
    evidence_ids: tuple[str, ...]
    key: str  # what it rests on: the same result seen again rests on the same key


def _digest(parts: Iterable[str]) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _content(items: Sequence[EvidenceItem]) -> str:
    """What the results showed, numbers aside: a timing or a count does not make a re-run new."""
    return _digest(sorted(f"{i.tool_name}|{_NUMBERS.sub('#', i.target)}|{i.status}|{_NUMBERS.sub('#', i.text[:300])}"
                          for i in items))


def reporter(item: EvidenceItem) -> tuple[str, str]:
    """Who a report comes from: a coding agent run from the shell, or a subagent."""
    if item.tool_name in _SHELL_TOOLS:
        found = _AGENT_NAME.search(item.target or "")
        return "external_agent", found.group(1) if found else "coding-agent"
    found = _MODEL_FIELD.search(item.text or "")
    return "subagent", f"subagent:{found.group(1)}" if found else "subagent"


def turn_outcomes(report: EvidenceReport, *, model: str, scope_key: str, turn_key: str) -> list[OutcomeDraft]:
    """One draft per claim class and actor for this turn, the worst reading winning. ``scope_key``
    is the task (or conversation) the claims belong to: a check seen again within it counts once.
    ``turn_key`` is this turn: a claim with nothing behind it counts once per turn."""
    by_id = {item.evidence_id: item for item in report.evidence}
    drafts: list[OutcomeDraft] = []
    for check in report.claims:
        items = [by_id[i] for i in check.evidence_ids if i in by_id]
        if check.reason == "only_historical_record" or (items and all(i.kind == "historical" for i in items)):
            continue  # an earlier session's record, not this turn's result
        if items and all(i.kind == "delegate_report" for i in items):
            continue  # a relayed report counts for whoever reported it, below
        result = "success" if check.supported else _SELF_RESULTS.get(check.reason, "unknown")
        key = (f"self:{scope_key}:{check.claim_class}:{_content(items)}" if items and result != "unknown"
               else f"self:{turn_key}:{check.claim_class}")
        drafts.append(OutcomeDraft(check.claim_class, "self", model or "unknown-model", result, check.reason,
                                   tuple(i.evidence_id for i in items), key))
    ours = own_verdicts(drafts)
    for item in report.evidence:
        if item.kind != "delegate_report":
            continue
        actor_kind, actor_id = reporter(item)
        if item.status != "ok":
            drafts.append(OutcomeDraft("task", actor_kind, actor_id, _REPORT_STATUS.get(item.status, "unknown"),
                                       item.status, (item.evidence_id,), f"report:{item.evidence_id}:task"))
            continue
        for claim_class in dict.fromkeys(claim.claim_class for claim in state_claims(item.text or "")):
            drafts.append(OutcomeDraft(claim_class, actor_kind, actor_id, ours.get(claim_class, "reported"),
                                       "checked" if claim_class in ours else "only_subagent_report",
                                       (item.evidence_id,), f"report:{item.evidence_id}:{claim_class}"))
    return _worst_per_slot(drafts)


def own_verdicts(drafts: Iterable[OutcomeDraft]) -> dict[str, str]:
    """What our own checks showed this turn, per claim class: success or failure (failure wins)."""
    found: dict[str, str] = {}
    for draft in drafts:
        if draft.actor_kind == "self" and draft.result in ("success", "failure") and found.get(draft.claim_class) != "failure":
            found[draft.claim_class] = draft.result
    return found


def _worst_per_slot(drafts: Iterable[OutcomeDraft]) -> list[OutcomeDraft]:
    chosen: dict[tuple[str, str, str], OutcomeDraft] = {}
    for draft in drafts:
        slot = (draft.claim_class, draft.actor_kind, draft.actor_id)
        if slot not in chosen or _WORST_FIRST.index(draft.result) < _WORST_FIRST.index(chosen[slot].result):
            chosen[slot] = draft
    return list(chosen.values())


def command_markers(behavior: str) -> tuple[str, ...]:
    """The commands a lesson names — in backticks, or a bare run of command words — cut to their
    first three words without flags, so `uv run pytest -q` is followed by `uv run pytest tests/`.
    A lesson in plain prose names none, and whether it was followed is not judged."""
    markers = [" ".join([w for w in text.split() if not w.startswith("-")][:3]).lower()
               for text in _BACKTICKED.findall(behavior or "")]
    if not any(markers):
        for run in _COMMAND_RUN.findall(behavior or ""):
            words = run.split()
            if {word.lower() for word in words} & _PROSE:
                continue
            words = [word for word in words if not word.startswith("-")][:3]
            if len(words) >= 2:
                markers.append(" ".join(words).lower())
    return tuple(dict.fromkeys(marker for marker in markers if marker))


def followed(markers: Sequence[str], evidence: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    """Our own results that ran one of the lesson's commands and showed it succeeded."""
    hits = []
    for item in evidence:
        if item.provenance != "self" or item.status != "ok" or not markers:
            continue
        shown = any(ok for _kind, ok in (item.steps or ((item.kind, True),)))
        call = " ".join(str(item.target or "").lower().split())
        if shown and any(marker in call for marker in markers):
            hits.append(item)
    return hits
