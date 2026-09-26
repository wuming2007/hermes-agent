"""Which state claims in an answer the current turn's tool results actually support.

Pure and deterministic (no IO, no model call). Every tool result of the turn becomes an
``EvidenceItem`` — what the call did (``kind``) and how it ended (``status``) — and every state
claim ``agent.cognition_claims`` finds in the answer becomes a ``ClaimCheck`` against evidence of
its own kind. A landed write is not a working feature, a test run before the last write is stale,
a timed-out or still-running command is not a result, and a subagent's report is not our own
check unless the answer attributes it. The evidence gate, the evidence-aware guard and the
process monitor all read the same ``EvidenceReport``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agent.cognition_claims import StateClaim, normalize_sentence, state_claims
from agent.display import _detect_tool_failure
from agent.tool_result_classification import FILE_MUTATING_TOOL_NAMES, NO_EFFECT_TOOL_NAMES
from tools.tool_search_catalog import TOOL_CALL_NAME
from tools.tool_search_validation import normalize_tool_call_entries
from utils import safe_json_loads

DIGEST_BUDGET = 2000
VERIFIER_EVIDENCE_BUDGET = 6000
_MAX_LISTED = 20
_EXCERPT_CHARS = 320
_ACTION_KINDS = frozenset({"write", "test", "exec", "deploy", "background_spawn", "delegate_report"})
_OBSERVATION_KINDS = frozenset({"read", "network", "exec", "test"})


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    tool_name: str
    kind: str  # read | write | exec | test | deploy | network | background_spawn | delegate_report
    status: str  # ok | error | blocked | timeout | running | partial
    target: str  # the path, command, URL or goal the call acted on
    provenance: str  # self | subagent
    index: int  # message index of the result row
    kinds: frozenset = frozenset()  # every kind the call covered: `pytest && ./deploy.sh` is test and deploy
    text: str = field(default="", repr=False, compare=False)  # the result as the model saw it


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    claim_class: str
    supported: bool
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()
    span: tuple[int, int] = (0, 0)  # where in ``text`` the claim is made


@dataclass(frozen=True)
class EvidenceReport:
    evidence: tuple[EvidenceItem, ...]
    claims: tuple[ClaimCheck, ...]
    digest: str = ""

    @property
    def unsupported(self) -> tuple[ClaimCheck, ...]:
        return tuple(c for c in self.claims if not c.supported)


# ---------------------------------------------------------------- evidence classification

_NETWORK_TOOLS = frozenset({"web_extract", "web_search", "browser_navigate"})
_READ_TOOLS = (NO_EFFECT_TOOL_NAMES | {"todo_list", "todo", "clarify", "task_state"}) - _NETWORK_TOOLS
_TOOL_KINDS = {
    **{name: "write" for name in FILE_MUTATING_TOOL_NAMES},
    **{name: "network" for name in _NETWORK_TOOLS},
    **{name: "read" for name in _READ_TOOLS},
    "delegate_task": "delegate_report",
}

# First match wins, so a test that also redirects output is still a test. Runner and deploy names
# count as the command word, not as part of a path (a tmp dir named pytest-of-ci is not a test).
_COMMAND_KINDS = (
    ("test", re.compile(
        r"(?<![\w./-])(?:pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|unittest)(?![\w-])"
        r"|(?:^|\s)\S*/(?:pytest|py\.test)(?=\s|$)|\brun_tests(?:\.sh)?\b"
        r"|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\b(?:go|cargo|mix|dotnet|swift|deno)\s+test\b"
        r"|\bmake\s+(?:test|check)\b|\bgh\s+(?:run\s+(?:view|watch)|pr\s+checks)\b")),
    ("deploy", re.compile(
        r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:(?:bash|sh|zsh|npm\s+run|pnpm\s+run|yarn|make|fly|vercel|netlify)\s+)?"
        r"(?:\S*/)?deploy[\w.-]*(?=\s|$)|\bgit\s+push\b|(?<![\w./-])(?:restart|reload)(?![\w-])"
        r"|\blaunchctl\s+(?:kickstart|bootstrap|load)\b|\bbrew\s+services\s+start\b"
        r"|\bdocker(?:\s+compose|-compose)?\s+(?:up|push)\b|\bkubectl\s+(?:apply|rollout|set|scale)\b"
        r"|\bhelm\s+(?:install|upgrade)\b|\b(?:npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b"
        r"|\bgh\s+release\s+create\b|\bterraform\s+apply\b")),
    ("write", re.compile(
        r"\bsed\s+-i|\btee\b|(?<![0-9&>])>>?\s*(?!/dev/null|&)[\w./~$\"'-]"
        r"|\b(?:cp|mv|mkdir|touch|rm|rmdir|ln|chmod|chown|truncate)\s"
        r"|\bgit\s+(?:commit|add|rm|mv|checkout|switch|reset|restore|apply|am|cherry-pick|merge|rebase|stash|tag)\b"
        r"|\b(?:npm|pnpm|yarn|pip|uv|brew)\s+(?:install|add|remove|uninstall|upgrade)\b")),
    ("network", re.compile(r"^(?:curl|wget|http|xh)\b")),
    ("read", re.compile(
        r"^(?:cat|ls|head|tail|less|grep|rg|find|fd|pwd|which|wc|stat|file|du|df|diff|echo|env|printenv|ps|tree"
        r"|git\s+(?:status|diff|log|show|branch|remote|rev-parse))\b")),
)
_LEADING_CD = re.compile(r"^\s*(?:cd\s+\S+\s*&&\s*)*")
_CHAIN = re.compile(r"\s*(?:&&|\|\||;)\s*")
_PROCESS_REPORT_ACTIONS = frozenset({"poll", "wait", "log"})
_PROCESS_ACTION_KINDS = {"list": "read"}


def command_kind(command: str) -> str:
    """What a shell command does, by its first matching step kind (exec when nothing matches)."""
    command = _LEADING_CD.sub("", command or "")
    return next((kind for kind, pattern in _COMMAND_KINDS if pattern.search(command)), "exec")


_STATE_CHANGING = tuple(pattern for kind, pattern in _COMMAND_KINDS if kind in ("deploy", "write"))


def command_changes_state(command: str) -> bool:
    """Whether any step of a shell command looks like a write or a deploy, whatever else it runs."""
    command = _LEADING_CD.sub("", command or "")
    return any(pattern.search(command) for pattern in _STATE_CHANGING)


def _command_kinds(command: str) -> frozenset:
    """The kind of the whole command plus the kind of each chained step."""
    steps = [step for step in _CHAIN.split(_LEADING_CD.sub("", command or "")) if step]
    return frozenset({command_kind(command), *(command_kind(step) for step in steps)})


def _error_text(data: dict, raw: str) -> str:
    return str(data.get("error") or "") if data else (raw if raw.startswith("Error") else "")


# Checked in order; the first that matches names the status (a refused approval that "timed out"
# is still a refusal, not a slow command).
_STATUS_RULES: tuple[tuple[str, Callable[[dict, str], bool]], ...] = (
    ("blocked", lambda d, raw: d.get("status") == "blocked" or _error_text(d, raw).startswith("BLOCKED")),
    ("timeout", lambda d, raw: d.get("exit_code") == 124 or "timed out" in _error_text(d, raw).lower()),
    ("running", lambda d, raw: d.get("status") in {"yielded_to_background", "running", "dispatched"}),
)
# delegate_task result entries; anything else (failed, error, interrupted) did not complete.
_SUBAGENT_STATUSES = {"completed": "ok", "timeout": "timeout"}


def _generic_status(tool_name: str, data: dict, raw: str) -> str:
    matched = next((status for status, rule in _STATUS_RULES if rule(data, raw)), None)
    if matched:
        return matched
    return "error" if _detect_tool_failure(tool_name, raw)[0] else "ok"


def _process_status(data: dict, raw: str) -> str:
    # A process_manage poll/wait reports on a background job: only an exited job is a result.
    if data.get("status") == "exited":
        return {0: "ok", 124: "timeout"}.get(data.get("exit_code"), "error")
    if data.get("status") in {"running", "timeout", "interrupted"}:
        return "running"
    return _generic_status("process_manage", data, raw)


def _results_status(data: dict, raw: str) -> str:
    entries = data.get("results")
    if not isinstance(entries, list) or not entries:
        return _generic_status("", data, raw)
    states = {_SUBAGENT_STATUSES.get(str(e.get("status")), "error") if "status" in e
              else ("error" if e.get("error") else "ok") for e in entries if isinstance(e, dict)}
    if len(states) > 1:
        return "partial"
    return states.pop() if states else "ok"


_STATUS_BY_TOOL = {
    "process_manage": _process_status,
    "web_extract": _results_status,
    "delegate_task": _results_status,
}


def _field(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _as_dict(value: Any) -> dict:
    parsed = safe_json_loads(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, dict) else {}


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(_field(part, "text") or "") for part in content)
    return "" if content is None else str(content)


def _unwrap(name: str, args: dict) -> tuple[str, dict]:
    """A ``tool_call`` bridge call with one entry is that tool's call."""
    if name != TOOL_CALL_NAME:
        return name, args
    entries, error = normalize_tool_call_entries(args)
    if error is None and len(entries) == 1:
        return str(entries[0].get("name") or name), _as_dict(entries[0].get("arguments"))
    return name, args


def _kinds(name: str, args: dict, data: dict, command: str) -> tuple[str, frozenset]:
    """The call's primary kind and every kind it covered. A process_manage poll/wait/log speaks
    for its job's command; kill/write/submit only act on the job."""
    report = name == "process_manage" and args.get("action") in _PROCESS_REPORT_ACTIONS
    if name == "terminal" and (args.get("background") or data.get("output") == "Background process started"):
        return "background_spawn", frozenset({"background_spawn"}) | _command_kinds(command)
    if name == "process_manage" and not report:
        kind = _PROCESS_ACTION_KINDS.get(str(args.get("action")), "exec")
    elif command and (name == "terminal" or report):
        return command_kind(command), _command_kinds(command)
    else:
        kind = _TOOL_KINDS.get(name, "exec")
    return kind, frozenset({kind})


def _classify(call_id: str, name: str, args: dict, raw: str, index: int) -> EvidenceItem:
    data = _as_dict(raw)
    command = str(args.get("command") or data.get("command") or "")
    kind, kinds = _kinds(name, args, data, command)
    status_of = _STATUS_BY_TOOL.get(name) if data else None
    status = status_of(data, raw) if status_of else _generic_status(name, data, raw)
    if kind == "background_spawn" and status == "ok":
        status = "running"  # a started job is not a result until it exits
    urls = args.get("urls")
    target = (args.get("path") or command or args.get("goal") or args.get("url")
              or (urls[0] if isinstance(urls, list) and urls else "") or args.get("query") or "")
    return EvidenceItem(
        evidence_id=f"e_{call_id}", tool_name=name, kind=kind, status=status, target=str(target),
        provenance="subagent" if kind == "delegate_report" else "self", index=index, kinds=kinds, text=raw,
    )


def classify_tool_result(tool_name: str, args: Any, result: Any, tool_call_id: str = "") -> EvidenceItem:
    """One tool result as evidence, for callers outside a message list (a ``post_tool_call`` hook)."""
    name, arguments = _unwrap(str(tool_name or ""), _as_dict(args))
    return _classify(str(tool_call_id or ""), name, arguments, _content_text(result), index=0)


# Background work reports back in user-role notices (tools/process_registry_notifications.py): a
# background job's exit, and an async subagent's result (top-level delegations always run async).
_JOB_NOTICE = re.compile(
    r"\A\[IMPORTANT: Background process (?P<sid>\S+) [^\n]*?\(exit code (?P<exit>-?\d+)[^)\n]*\)\.\n"
    r"(?:[^\n]*\n)??Command: (?P<cmd>[^\n]*)")
_DELEGATION_NOTICE = re.compile(r"\A\[ASYNC DELEGATION (?:BATCH )?COMPLETE — (?P<sid>[^\]\s]+)\]")
_NOTICE_GOAL = re.compile(r"^Original goal: (?P<goal>.*)$", re.MULTILINE)
_NOTICE_STATUS = re.compile(r"^Status: (?P<status>\w+)", re.MULTILINE)


def _job_notice(match: re.Match, text: str, index: int) -> EvidenceItem:
    command = match["cmd"].strip()
    return EvidenceItem(
        evidence_id=f"e_notice_{match['sid']}_{index}", tool_name="process_notice", kind=command_kind(command),
        status={0: "ok", 124: "timeout"}.get(int(match["exit"]), "error"), target=command, provenance="self",
        index=index, kinds=_command_kinds(command), text=text,
    )


def _delegation_notice(match: re.Match, text: str, index: int) -> EvidenceItem:
    states = {_SUBAGENT_STATUSES.get(m["status"], "error") for m in _NOTICE_STATUS.finditer(text)} or {"error"}
    goal = _NOTICE_GOAL.search(text)
    return EvidenceItem(
        evidence_id=f"e_delegation_{match['sid']}_{index}", tool_name="delegation_notice", kind="delegate_report",
        status=states.pop() if len(states) == 1 else "partial", target=goal["goal"].strip() if goal else "",
        provenance="subagent", index=index, kinds=frozenset({"delegate_report"}), text=text,
    )


_NOTICES: tuple[tuple[re.Pattern, Callable[[re.Match, str, int], EvidenceItem]], ...] = (
    (_JOB_NOTICE, _job_notice),
    (_DELEGATION_NOTICE, _delegation_notice),
)


def _notice_item(text: str, index: int) -> EvidenceItem | None:
    for pattern, build in _NOTICES:
        match = pattern.match(text)
        if match is not None:
            return build(match, text, index)
    return None


def _is_turn_start(message: Any) -> bool:
    """A user row carrying the user's own intent; runtime nudges and notices continue the turn."""
    from agent.conversation_compression import _is_real_user_message

    return _is_real_user_message(message) and not any(
        str(key).endswith("_synthetic") and value for key, value in message.items())


def classify_turn_evidence(messages: Sequence[Any], turn_start: int | None = None) -> tuple[EvidenceItem, ...]:
    """Classify every tool result (and background-job notice) since the turn's user row."""
    if turn_start is None:
        turn_start = next((i for i in range(len(messages) - 1, -1, -1) if _is_turn_start(messages[i])), 0)
    calls: dict[str, tuple[str, dict]] = {}
    items: list[EvidenceItem] = []
    for index in range(turn_start, len(messages)):
        message = messages[index]
        for call in _field(message, "tool_calls") or ():
            fn = _field(call, "function")
            name, args = _unwrap(str(_field(fn, "name") or ""), _as_dict(_field(fn, "arguments")))
            calls[str(_field(call, "id") or "")] = (name, args)
        role, call_id = _field(message, "role"), str(_field(message, "tool_call_id") or "")
        text = _content_text(_field(message, "content"))
        if role == "tool" and call_id in calls:
            items.append(_classify(call_id, *calls[call_id], text, index))
        elif role == "user" and (notice := _notice_item(text, index)) is not None:
            items.append(notice)
    return tuple(items)


# ---------------------------------------------------------------- claims against evidence


@dataclass(frozen=True)
class _Support:
    kinds: frozenset  # evidence kinds that can support the claim
    after_last_write: bool  # support must postdate the last landed write
    missing: str  # reason when nothing of those kinds exists
    stale: str = ""  # reason when there was a write and nothing checked it since
    any_landed: bool = False  # one ok item suffices (writes); otherwise the latest item decides
    universal: bool = False  # "all done": any unfinished action contradicts it


SUPPORT = {
    "deployed": _Support(frozenset({"deploy"}), False, "no_deploy_evidence"),
    "tests_pass": _Support(frozenset({"test"}), True, "no_test_evidence", "no_test_after_last_write"),
    "verified": _Support(frozenset({"test", "exec", "network", "read", "deploy"}), True,
                         "no_check_evidence", "no_check_after_last_write"),
    "working": _Support(frozenset({"test", "exec", "network"}), True,
                        "no_functional_check", "no_functional_check_after_write"),
    "written": _Support(frozenset({"write"}), False, "no_write_evidence", any_landed=True),
    "completed": _Support(_ACTION_KINDS, False, "no_action_evidence", universal=True),
}
_SHELL_TOOLS = {"terminal": "shell", "process_manage": "shell", "process_notice": "shell"}
_STATUS_REASONS = {"timeout": "timeout", "running": "still_running", "error": "failed", "blocked": "blocked",
                   "partial": "partial"}
_Verdict = tuple[bool, str, tuple[str, ...]]


def _subagent_verdict(claim: StateClaim, reports: list[EvidenceItem]) -> _Verdict:
    last = reports[-1]
    if last.status != "ok":
        return False, _STATUS_REASONS[last.status], (last.evidence_id,)
    return (True, "", (last.evidence_id,)) if claim.attributed else (False, "only_subagent_report", (last.evidence_id,))


def _check_universal(claim: StateClaim, items: Sequence[EvidenceItem]) -> _Verdict:
    latest = {(_SHELL_TOOLS.get(e.tool_name, e.tool_name), e.target): e for e in items if e.kind in _ACTION_KINDS}
    for status in ("timeout", "running", "error", "blocked", "partial"):
        stuck = [e.evidence_id for e in latest.values() if e.status == status]
        if stuck:
            return False, _STATUS_REASONS[status], tuple(stuck)
    done = [e.evidence_id for e in items if e.kind in _ACTION_KINDS and e.provenance == "self" and e.status == "ok"]
    if done:
        return True, "", tuple(done[-5:])
    reports = [e for e in items if e.provenance == "subagent"]
    return _subagent_verdict(claim, reports) if reports else (False, "no_action_evidence", ())


def _check(claim: StateClaim, items: Sequence[EvidenceItem]) -> _Verdict:
    support = SUPPORT[claim.claim_class]
    if support.universal:
        return _check_universal(claim, items)
    last_write = max((e.index for e in items if e.kind == "write" and e.status == "ok"), default=None)
    candidates = [
        e for e in items if e.provenance == "self" and e.kinds & support.kinds
        and (not support.after_last_write or last_write is None or e.index > last_write)
    ]
    if candidates:
        landed = [e for e in candidates if e.status == "ok"]
        chosen = landed[-1] if support.any_landed and landed else candidates[-1]
        reason = "" if chosen.status == "ok" else _STATUS_REASONS[chosen.status]
        return chosen.status == "ok", reason, (chosen.evidence_id,)
    reports = [e for e in items if e.provenance == "subagent"]
    if reports:
        return _subagent_verdict(claim, reports)
    stale = support.after_last_write and last_write is not None
    return False, (support.stale if stale else support.missing), ()


def check_claims(answer: str, evidence: Sequence[EvidenceItem]) -> tuple[ClaimCheck, ...]:
    """Judge every state claim of the answer against evidence of its own kind."""
    return tuple(ClaimCheck(c.sentence, c.claim_class, *_check(c, evidence), span=c.span)
                 for c in state_claims(answer))


def evidence_refs_for(sentence: str, report: EvidenceReport) -> tuple[str, ...]:
    """Evidence ids behind one sentence of the reported answer: its state claims' own evidence
    (none if any is unsupported), or for a plain factual sentence the turn's successful
    observations. Claims are looked up in the whole-answer report, where an attribution in the
    sentence before still counts."""
    key = normalize_sentence(sentence)
    checks = [c for c in report.claims if c.text == key]
    if checks:
        return tuple(dict.fromkeys(i for c in checks for i in c.evidence_ids)) if all(c.supported for c in checks) else ()
    return tuple(e.evidence_id for e in report.evidence
                 if e.provenance == "self" and e.status == "ok" and e.kind in _OBSERVATION_KINDS)[-5:]


# ---------------------------------------------------------------- report

_KIND_PRIORITY = {"write": 0, "test": 0, "deploy": 0, "background_spawn": 0, "delegate_report": 0,
                  "exec": 1, "network": 1, "read": 2}
_TOKEN = re.compile(r"[A-Za-z0-9_][\w.\-/]{2,}", re.ASCII)
_CONTEXT = 60  # characters kept on each side of a word the answer relies on


def _answer_tokens(answer: str) -> list[str]:
    """File names, identifiers and numbers the answer mentions — what a result must show."""
    return list(dict.fromkeys(t.lower() for t in _TOKEN.findall(answer or "")))[:30]


def _excerpt(text: str, tokens: Sequence[str]) -> str:
    """What the result said: the passages around words the answer relies on, else head and tail."""
    flat = " ".join(text.split())
    if len(flat) <= _EXCERPT_CHARS:
        return flat
    lower = flat.lower()
    spans: list[list[int]] = []
    for token in tokens:
        at = lower.find(token)
        if at >= 0:
            spans.append([max(0, at - _CONTEXT), min(len(flat), at + len(token) + _CONTEXT)])
    if not spans:
        head = _EXCERPT_CHARS * 2 // 3
        return f"{flat[:head]} … {flat[-(_EXCERPT_CHARS - head - 3):]}"
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return _clip(" … ".join(flat[start:end] for start, end in merged), 2 * _EXCERPT_CHARS)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def claim_excerpt(check: ClaimCheck, limit: int = 80) -> str:
    """The claim's sentence cut to ``limit`` around the words that make the claim, so a long
    sentence never shows its backed clauses and hides the unbacked one."""
    text = check.text
    if len(text) <= limit:
        return text
    start, end = check.span
    lo = max(0, min(start - max(0, (limit - (end - start)) // 2), len(text) - limit))
    hi = min(len(text), lo + limit)
    return ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def _rank(evidence: Sequence[EvidenceItem], tokens: Sequence[str]) -> list[EvidenceItem]:
    """Failures first, then results that mention what the answer mentions, then the kinds that
    bear on state claims, newest first."""

    def relevance(item: EvidenceItem) -> int:
        seen = f"{item.target} {item.text}".lower()
        return sum(token in seen for token in tokens)

    return sorted(evidence, key=lambda e: (e.status == "ok", -relevance(e), _KIND_PRIORITY.get(e.kind, 1), -e.index))


def _render(evidence: Sequence[EvidenceItem], checks: Sequence[ClaimCheck], answer: str, *,
            budget: int, excerpts: bool) -> str:
    gaps = [f"- [{c.claim_class}] {claim_excerpt(c)} → {c.reason}"
            + (f" ({', '.join(c.evidence_ids[:3])})" if c.evidence_ids else "") for c in checks if not c.supported]
    tail = (["Unsupported claims:", *gaps] if gaps else
            ["All state claims have evidence of their kind."] if checks else [])
    header = f"Evidence this turn ({len(evidence)} tool results):" if evidence else "No tool results this turn."
    used = len(header) + sum(len(line) + 1 for line in tail) + 60  # 60: the omitted-count line
    tokens = _answer_tokens(answer)
    chosen: list[tuple[int, str]] = []
    for item in _rank(evidence, tokens)[:_MAX_LISTED]:
        line = f"- {item.evidence_id} {item.tool_name} {item.kind} {item.status}: {_clip(item.target, 70)}"
        line += f"\n  → {_excerpt(item.text, tokens)}" if excerpts and item.text.strip() else ""
        if used + len(line) + 1 > budget:
            break
        chosen.append((item.index, line))
        used += len(line) + 1
    omitted = len(evidence) - len(chosen)
    lines = [header, *(line for _, line in sorted(chosen))]
    lines += [f"- … {omitted} more results omitted"] if omitted else []
    return "\n".join([*lines, *tail])[:budget]


def build_evidence_report(answer: str, messages: Sequence[Any], turn_start: int | None = None) -> EvidenceReport:
    """Classify the turn's tool results and check the answer's state claims against them."""
    evidence = classify_turn_evidence(messages, turn_start)
    checks = check_claims(answer, evidence)
    return EvidenceReport(evidence, checks, _render(evidence, checks, answer, budget=DIGEST_BUDGET, excerpts=False))


def evidence_for_verifier(answer: str, messages: Sequence[Any]) -> str:
    """The turn's evidence with what each result said, for a verifier that has to tell facts read
    off a tool result from invented ones; results the answer relies on come first."""
    evidence = classify_turn_evidence(messages)
    checks = check_claims(answer, evidence)
    return _render(evidence, checks, answer, budget=VERIFIER_EVIDENCE_BUDGET, excerpts=True)


def evidence_metadata(report: EvidenceReport) -> dict[str, Any]:
    """Flat cognition metadata for the trace and autonomy telemetry."""
    return {
        "evidence_tool_result_count": len(report.evidence),
        "evidence_claim_count": len(report.claims),
        "evidence_supported_count": sum(c.supported for c in report.claims),
        "evidence_unsupported": [{"class": c.claim_class, "reason": c.reason, "text": claim_excerpt(c, 200)}
                                 for c in report.unsupported],
        "tool_evidence_present": any(e.provenance == "self" and e.status == "ok" for e in report.evidence),
    }
