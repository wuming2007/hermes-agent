"""How this turn's claimed results turned out, and which lessons were followed (cognition v2, phase 2).

Pure: the turn's evidence report goes in, outcome drafts come out. Our own claims count by what
stands behind them: a backed claim is a success; a failed, blocked or partial step is exactly that;
anything short of a shown result is unknown. A subagent's or coding agent's report counts for
whoever reported it, as reported, until our own check of what it names confirms or refutes it. An
earlier session's record is not this turn's result and counts for nothing. The same check seen again
(a retry, a re-run whose output differs only in its timing) rests on the same key, so it counts
once; the same command on another file, port or id is another check.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from agent.cognition_claims import state_claims
from agent.cognition_evidence import EvidenceItem, EvidenceReport, check_claims
from agent.command_effects import executed_commands, invokes
from plugins.cognition_wm.lessons import avoided

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
# What a rerun shows differently without it being another result: timings, clock times, addresses.
_VOLATILE = (
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"), "<date>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<time>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|msec|s|secs?|seconds?|mins?|minutes?)\b", re.IGNORECASE), "<duration>"),
    (re.compile(r"\b0x[0-9a-f]{6,}\b", re.IGNORECASE), "<address>"),
)
_WORKDIR = re.compile(r" \(in [^()]*\)$")
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
    claim: str = ""  # a report's own words for it: what our later checks must be about to settle it


def _digest(parts: Iterable[str]) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _steady(text: str) -> str:
    for pattern, mark in _VOLATILE:
        text = pattern.sub(mark, text)
    return text


def _content(items: Sequence[EvidenceItem]) -> str:
    """What the results showed: the same command on the same target with the same output. A timing
    does not make a re-run new; another file, port or id makes it another check."""
    return _digest(sorted(f"{i.tool_name}|{i.target}|{i.status}|{_steady(i.text[:300])}" for i in items))


def _said(item: EvidenceItem) -> str:
    """A report in its own words, without the tool result's wrapping (``{"output": ..., "exit_code": 0}``,
    a subagent's ``results``): the wrapping's keys are not what the report is about."""
    try:
        data = json.loads(item.text or "")
    except ValueError:
        return item.text or ""
    results = data.get("results") if isinstance(data, dict) else None
    found = [entry[key] for entry in (data, *(results if isinstance(results, list) else ())) if isinstance(entry, dict)
             for key in ("output", "summary", "result", "response") if isinstance(entry.get(key), str)]
    return "\n".join(found) if found else item.text or ""


def reporter(item: EvidenceItem) -> tuple[str, str]:
    """Who a report comes from: a coding agent run from the shell, or a subagent."""
    if item.tool_name in _SHELL_TOOLS:
        found = _AGENT_NAME.search(item.target or "")
        return "external_agent", found.group(1) if found else "coding-agent"
    found = _MODEL_FIELD.search(item.text or "")
    return "subagent", f"subagent:{found.group(1)}" if found else "subagent"


def turn_outcomes(report: EvidenceReport, *, model: str, scope_key: str, turn_key: str,
                  actor: str = "") -> list[OutcomeDraft]:
    """One draft per claim class and actor for this turn, the worst reading winning. ``scope_key``
    is the task (or conversation) the claims belong to, and ``actor`` who made them (model, provider
    and Hermes version): a check seen again within it by the same actor counts once, and another
    model's rerun is that model's own record. ``turn_key`` is this turn: a claim with nothing behind
    it counts once per turn. A report is settled only by our checks after it."""
    by_id = {item.evidence_id: item for item in report.evidence}
    drafts: list[OutcomeDraft] = []
    for check in report.claims:
        items = [by_id[i] for i in check.evidence_ids if i in by_id]
        if check.reason == "only_historical_record" or (items and all(i.kind == "historical" for i in items)):
            continue  # an earlier session's record, not this turn's result
        if items and all(i.kind == "delegate_report" for i in items):
            continue  # a relayed report counts for whoever reported it, below
        result = "success" if check.supported else _SELF_RESULTS.get(check.reason, "unknown")
        key = (f"self:{scope_key}:{actor}:{check.claim_class}:{_content(items)}" if items and result != "unknown"
               else f"self:{turn_key}:{check.claim_class}")
        drafts.append(OutcomeDraft(check.claim_class, "self", model or "unknown-model", result, check.reason,
                                   tuple(i.evidence_id for i in items), key))
    for item in report.evidence:
        if item.kind != "delegate_report":
            continue
        actor_kind, actor_id = reporter(item)
        if item.status != "ok":
            drafts.append(OutcomeDraft("task", actor_kind, actor_id, _REPORT_STATUS.get(item.status, "unknown"),
                                       item.status, (item.evidence_id,), f"report:{item.evidence_id}:task"))
            continue
        said: dict[str, list[str]] = {}
        for claim in state_claims(_said(item)):
            said.setdefault(claim.claim_class, []).append(claim.sentence)
        for claim_class, sentences in said.items():
            words = " ".join(sentences)[:300]
            result, _ids = settle_report(words, [later for later in report.evidence if later.index > item.index])
            drafts.append(OutcomeDraft(claim_class, actor_kind, actor_id, result,
                                       "checked" if result != "reported" else "only_subagent_report",
                                       (item.evidence_id,), f"report:{item.evidence_id}:{claim_class}", claim=words))
    return _worst_per_slot(drafts)


def settle_report(claim: str, evidence: Sequence[EvidenceItem]) -> tuple[str, tuple[str, ...]]:
    """What our own checks among ``evidence`` — the results that came after the report — say about a
    subagent's or coding agent's report (``claim``, its own words), by the rule our own claims are held
    to: a check counts only when it is about what the report names, each of them. Success when every
    claim holds, failure when a check of what it names failed, else it stays reported. Returns the
    result and our checks' ids."""
    own = {item.evidence_id for item in evidence if item.provenance == "self" and item.kind != "delegate_report"}
    checks = check_claims(claim, evidence)
    backed = [check for check in checks if check.evidence_ids and set(check.evidence_ids) <= own]
    failed = [check for check in backed if not check.supported and check.reason == "failed"]
    if failed:
        return "failure", failed[0].evidence_ids
    if checks and len(backed) == len(checks) and all(check.supported for check in backed):
        return "success", tuple(dict.fromkeys(i for check in backed for i in check.evidence_ids))
    return "reported", ()


def _worst_per_slot(drafts: Iterable[OutcomeDraft]) -> list[OutcomeDraft]:
    chosen: dict[tuple[str, str, str], OutcomeDraft] = {}
    for draft in drafts:
        slot = (draft.claim_class, draft.actor_kind, draft.actor_id)
        if slot not in chosen or _WORST_FIRST.index(draft.result) < _WORST_FIRST.index(chosen[slot].result):
            chosen[slot] = draft
    return list(chosen.values())


def command_markers(behavior: str) -> tuple[str, ...]:
    """The commands a lesson says to use — in backticks, or a bare run of command words — cut to
    their first three words without flags, so `uv run pytest -q` is followed by `uv run pytest tests/`.
    A command it says to avoid ("instead of `pytest`", "never use `X`") is not one. A lesson in plain
    prose names none, and whether it was followed is not judged."""
    text = behavior or ""
    found = list(_BACKTICKED.finditer(text))
    if not found:
        found = [m for m in _COMMAND_RUN.finditer(text)
                 if not {word.lower() for word in m.group(0).split()} & _PROSE]
    markers = []
    for match, avoid in zip(found, avoided([m.span() for m in found], text)):
        words = [word for word in match.group(match.lastindex or 0).split() if not word.startswith("-")][:3]
        if not avoid and (words if match.re is _BACKTICKED else len(words) >= 2):
            markers.append(" ".join(words).lower())
    return tuple(dict.fromkeys(markers))


def followed(markers: Sequence[str], evidence: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    """Our own shell results in which one of the lesson's commands itself ran, and the line's exit
    status shows that it succeeded. A command only printed or searched for (``echo 'uv run pytest'``)
    was not run."""
    hits = []
    for item in evidence:
        if item.provenance != "self" or item.status != "ok" or item.tool_name not in _SHELL_TOOLS or not markers:
            continue
        steps = executed_commands(_WORKDIR.sub("", str(item.target or "")))
        if any(shown and any(invokes(step, marker) for marker in markers) for step, shown in steps):
            hits.append(item)
    return hits
