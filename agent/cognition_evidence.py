"""Which state claims in an answer the current turn's tool results actually support.

Pure and deterministic (no IO, no model call). Every tool result of the turn becomes an
``EvidenceItem`` — what the call did (``kind``) and how it ended (``status``) — and every sentence
that claims a state ("tests pass", "deployed", "fixed", "done") becomes a ``ClaimCheck`` against
evidence of its own kind. A landed write is not a working feature, a test run before the last write
is stale, a timed-out or still-running command is not a result, and a subagent's report is not our
own check unless the answer attributes it. The evidence gate, the evidence-aware guard and the
process monitor all read the same ``EvidenceReport``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agent.display import _detect_tool_failure
from agent.process_monitor import extract_claims_from_response
from agent.tool_result_classification import FILE_MUTATING_TOOL_NAMES, NO_EFFECT_TOOL_NAMES
from tools.tool_search_catalog import TOOL_CALL_NAME
from tools.tool_search_validation import normalize_tool_call_entries
from utils import safe_json_loads

DIGEST_BUDGET = 2000
_DIGEST_MAX_ITEMS = 20
_ACTION_KINDS = frozenset({"write", "test", "exec", "deploy", "background_spawn", "delegate_report"})


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    tool_name: str
    kind: str  # read | write | exec | test | deploy | network | background_spawn | delegate_report
    status: str  # ok | error | blocked | timeout | running | partial
    target: str  # the path, command, URL or goal the call acted on
    provenance: str  # self | subagent
    index: int  # message index of the tool result row


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    claim_class: str
    supported: bool
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()


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

# First match wins, so a test that also redirects output is still a test.
_COMMAND_KINDS = (
    ("test", re.compile(
        r"\b(?:pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|unittest|run_tests)\b"
        r"|\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\b(?:go|cargo|mix|dotnet|swift|deno)\s+test\b"
        r"|\bmake\s+(?:test|check)\b|\bgh\s+(?:run\s+(?:view|watch)|pr\s+checks)\b")),
    ("deploy", re.compile(
        r"\bdeploy\w*|\bgit\s+push\b|\brestart\b|\breload\b"
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
_PROCESS_REPORT_ACTIONS = frozenset({"poll", "wait", "log"})
_PROCESS_ACTION_KINDS = {"list": "read"}


def _command_kind(command: str) -> str:
    command = _LEADING_CD.sub("", command or "")
    return next((kind for kind, pattern in _COMMAND_KINDS if pattern.search(command)), "exec")


def _error_text(data: dict, raw: str) -> str:
    return str(data.get("error") or "") if data else (raw if raw.startswith("Error") else "")


# Checked in order; the first that matches names the status (a refused approval that "timed out"
# is still a refusal, not a slow command).
_STATUS_RULES: tuple[tuple[str, Callable[[dict, str], bool]], ...] = (
    ("blocked", lambda d, raw: d.get("status") == "blocked" or _error_text(d, raw).startswith("BLOCKED")),
    ("timeout", lambda d, raw: d.get("exit_code") == 124 or "timed out" in _error_text(d, raw).lower()),
    ("running", lambda d, raw: d.get("status") in {"yielded_to_background", "running", "dispatched"}),
)
_SUBAGENT_STATUSES = {"completed": "ok", "timeout": "timeout", "running": "running", "dispatched": "running"}


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


def _generic_status(tool_name: str, data: dict, raw: str) -> str:
    matched = next((status for status, rule in _STATUS_RULES if rule(data, raw)), None)
    if matched:
        return matched
    return "error" if _detect_tool_failure(tool_name, raw)[0] else "ok"


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


def _kind(name: str, args: dict, data: dict, command: str) -> str:
    if name == "terminal" and (args.get("background") or data.get("output") == "Background process started"):
        return "background_spawn"
    if name == "process_manage" and args.get("action") not in _PROCESS_REPORT_ACTIONS:
        return _PROCESS_ACTION_KINDS.get(str(args.get("action")), "exec")
    if name in ("terminal", "process_manage") and command:
        return _command_kind(command)
    return _TOOL_KINDS.get(name, "exec")


def _classify(call_id: str, name: str, args: dict, raw: str, index: int) -> EvidenceItem:
    data = _as_dict(raw)
    command = str(args.get("command") or data.get("command") or "")
    kind = _kind(name, args, data, command)
    status_of = _STATUS_BY_TOOL.get(name) if data else None
    if kind == "background_spawn":
        status = "running"  # a spawn only proves the job started
    else:
        status = status_of(data, raw) if status_of else _generic_status(name, data, raw)
    urls = args.get("urls")
    target = (args.get("path") or command or args.get("goal") or args.get("url")
              or (urls[0] if isinstance(urls, list) and urls else "") or args.get("query") or "")
    return EvidenceItem(
        evidence_id=f"e_{call_id}", tool_name=name, kind=kind, status=status, target=str(target),
        provenance="subagent" if kind == "delegate_report" else "self", index=index,
    )


def _is_turn_start(message: Any) -> bool:
    return _field(message, "role") == "user" and not any(
        str(key).endswith("_synthetic") and value for key, value in (message.items() if isinstance(message, dict) else ()))


def classify_turn_evidence(messages: Sequence[Any], turn_start: int | None = None) -> tuple[EvidenceItem, ...]:
    """Classify every tool result after the turn's user row (default: the last real user row)."""
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
        call_id = str(_field(message, "tool_call_id") or "")
        if _field(message, "role") == "tool" and call_id in calls:
            name, args = calls[call_id]
            items.append(_classify(call_id, name, args, _content_text(_field(message, "content")), index))
    return tuple(items)


# ---------------------------------------------------------------- claims

_CJK_DONE = r"(?:完成|好了|成功|完畢|了)"
_CJK_SINCE = r"已經?[^，,。；;！!？?\n]{0,6}?"  # 已／已經 plus a short object: 已經幫你寄出
_NEGATION = re.compile(
    r"尚未|還沒|還未|沒有|並未|未能|無法|不能|不會|不是|是否|等待|待|沒|未|不"
    r"|\b(?:not|never|no|yet to|unable|cannot|can't|without|whether)\b|n't\b", re.IGNORECASE)
_ATTRIBUTION = re.compile(r"回報|表示|聲稱|指出|據|說|\b(?:reports?|reported|says|said|claims?|according to)\b",
                          re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[，,。；;！!？?：:\n]")
_NEGATION_WINDOW = 16


@dataclass(frozen=True)
class _ClaimRule:
    claim_class: str
    pattern: re.Pattern
    kinds: frozenset  # evidence kinds that can support it
    after_last_write: bool  # support must postdate the last landed write
    missing: str  # reason when nothing of those kinds exists
    stale: str = ""  # reason when there was a write and nothing checked it since
    any_landed: bool = False  # one ok item suffices (writes); otherwise the latest item decides
    universal: bool = False  # "all done": any unfinished action contradicts it


# Precedence order: a sentence is checked as its strongest claim.
_CLAIM_RULES = (
    _ClaimRule("deployed", re.compile(
        rf"{_CJK_SINCE}(?:部署|佈署|上線|重啟|重新啟動|發布|發佈|推送|推上)|(?:部署|佈署|上線|重啟|重新啟動|發布|發佈){_CJK_DONE}"
        r"|\b(?:re)?deployed\b|\brestarted\b|\breleased\b|\brolled out\b|\bshipped\b|\bpushed\b|\b(?:is|now|went) live\b",
        re.IGNORECASE), frozenset({"deploy"}), False, "no_deploy_evidence"),
    _ClaimRule("tests_pass", re.compile(
        r"測試[^，,。；;\n]{0,12}?(?:通過|過了|全過|全綠)|\bCI\b[^，,。；;\n]{0,8}?(?:通過|綠)"
        r"|\b(?:tests?|specs?|CI|checks?)\b[^.,;\n]{0,20}?\b(?:pass(?:ed|es|ing)?|green|succeed(?:ed|s)?)\b",
        re.IGNORECASE), frozenset({"test"}), True, "no_test_evidence", "no_test_after_last_write"),
    _ClaimRule("verified", re.compile(
        r"已經?(?:驗證|核對)|(?:驗證|確認|核對|檢查)(?:過了|過|完成|無誤|正常|通過)"
        r"|\b(?:verified|confirmed|double-checked|validated)\b", re.IGNORECASE),
        frozenset({"test", "exec", "network", "read", "deploy"}), True, "no_check_evidence", "no_check_after_last_write"),
    _ClaimRule("working", re.compile(
        r"修(?:好|復|復好)|(?:可以|能夠?)正常(?:使用|運作|運行|執行|啟動|工作)|正常運作|運作正常|恢復正常|已經?解決|解決了"
        r"|\b(?:works|fixed|resolved)\b|\bis working\b", re.IGNORECASE),
        frozenset({"test", "exec", "network"}), True, "no_functional_check", "no_functional_check_after_write"),
    _ClaimRule("written", re.compile(
        rf"{_CJK_SINCE}(?:寫入|寫好|建立|新增|更新|修改|改好|儲存|存檔|存好|產生|生成|加入)|完成(?:了)?(?:修改|寫入|建立|新增|更新|編輯)"
        rf"|(?:寫入|建立|新增|儲存|產生|生成){_CJK_DONE}|\b(?:wrote|written|saved|created|updated|modified|edited)\b",
        re.IGNORECASE), frozenset({"write"}), False, "no_write_evidence", any_landed=True),
    _ClaimRule("completed", re.compile(
        rf"{_CJK_SINCE}(?:完成|搞定|做完|處理好|寄出|發送|送出|寄送|傳送|通知)|(?:全部|都|均)(?:已經?)?(?:完成|搞定|做完)"
        r"|(?:完成|搞定|做完)了|\b(?:done|completed|finished|all set|sent)\b", re.IGNORECASE),
        _ACTION_KINDS, False, "no_action_evidence", universal=True),
)


def _claim_match(rule: _ClaimRule, sentence: str) -> bool:
    for match in rule.pattern.finditer(sentence):
        breaks = [m.end() for m in _CLAUSE_BREAK.finditer(sentence, 0, match.start())]
        start = max(breaks[-1] if breaks else 0, match.start() - _NEGATION_WINDOW)
        if not _NEGATION.search(sentence[start:match.end()]):
            return True
    return False


_SHELL_TOOLS = {"terminal": "shell", "process_manage": "shell"}
_STATUS_REASONS = {"timeout": "timeout", "running": "still_running", "error": "failed", "blocked": "blocked",
                   "partial": "partial"}


def _subagent_verdict(sentence: str, reports: list[EvidenceItem]) -> tuple[bool, str]:
    last = reports[-1]
    if last.status != "ok":
        return False, _STATUS_REASONS[last.status]
    return (True, "") if _ATTRIBUTION.search(sentence) else (False, "only_subagent_report")


def _check_universal(sentence: str, items: Sequence[EvidenceItem]) -> tuple[bool, str, tuple[str, ...]]:
    latest = {(_SHELL_TOOLS.get(e.tool_name, e.tool_name), e.target): e for e in items if e.kind in _ACTION_KINDS}
    for status in ("timeout", "running", "error", "blocked", "partial"):
        stuck = [e.evidence_id for e in latest.values() if e.status == status]
        if stuck:
            return False, _STATUS_REASONS[status], tuple(stuck)
    done = [e.evidence_id for e in items if e.kind in _ACTION_KINDS and e.provenance == "self" and e.status == "ok"]
    if done:
        return True, "", tuple(done[-5:])
    reports = [e for e in items if e.provenance == "subagent"]
    if reports:
        return (*_subagent_verdict(sentence, reports), (reports[-1].evidence_id,))
    return False, "no_action_evidence", ()


def _check(rule: _ClaimRule, sentence: str, items: Sequence[EvidenceItem]) -> tuple[bool, str, tuple[str, ...]]:
    if rule.universal:
        return _check_universal(sentence, items)
    last_write = max((e.index for e in items if e.kind == "write" and e.status == "ok"), default=None)
    candidates = [
        e for e in items if e.provenance == "self" and (
            e.kind in rule.kinds or (e.kind == "background_spawn" and _command_kind(e.target) in rule.kinds))
        and (not rule.after_last_write or last_write is None or e.index > last_write)
    ]
    if candidates:
        landed = [e for e in candidates if e.status == "ok"]
        chosen = landed[-1] if rule.any_landed and landed else candidates[-1]
        verdict = (True, "") if chosen.status == "ok" else (False, _STATUS_REASONS[chosen.status])
        return (*verdict, (chosen.evidence_id,))
    reports = [e for e in items if e.provenance == "subagent"]
    if reports:
        return (*_subagent_verdict(sentence, reports), (reports[-1].evidence_id,))
    stale = rule.after_last_write and last_write is not None
    return False, (rule.stale if stale else rule.missing), ()


_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_CJK_SENTENCE_END = re.compile(r"(?<=[。！？；])")
_MAX_CHECKS = 16


def _sentences(answer: str) -> list[str]:
    """Answer sentences, code excluded; CJK sentences end at 。！？； without a following space."""
    claims = extract_claims_from_response(_CODE.sub(" ", answer or ""), max_claims=4 * _MAX_CHECKS)
    return [part.strip() for claim in claims for part in _CJK_SENTENCE_END.split(claim.text) if part.strip()]


def check_claims(answer: str, evidence: Sequence[EvidenceItem]) -> tuple[ClaimCheck, ...]:
    """Check each state-claiming sentence (questions and offers are not claims) against the evidence."""
    checks: list[ClaimCheck] = []
    for sentence in _sentences(answer):
        rule = None if sentence.endswith(("?", "？")) else next(
            (r for r in _CLAIM_RULES if _claim_match(r, sentence)), None)
        if rule is not None:
            supported, reason, ids = _check(rule, sentence, evidence)
            checks.append(ClaimCheck(sentence, rule.claim_class, supported, reason, ids))
    return tuple(checks[:_MAX_CHECKS])


# ---------------------------------------------------------------- report

_KIND_PRIORITY = {"write": 0, "test": 0, "deploy": 0, "background_spawn": 0, "delegate_report": 0,
                  "exec": 1, "network": 1, "read": 2}


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _digest(evidence: Sequence[EvidenceItem], checks: Sequence[ClaimCheck]) -> str:
    gaps = [f"- [{c.claim_class}] {_clip(c.text, 80)} → {c.reason}"
            + (f" ({', '.join(c.evidence_ids[:3])})" if c.evidence_ids else "") for c in checks if not c.supported]
    tail = (["Unsupported claims:", *gaps] if gaps else
            ["All state claims have evidence of their kind."] if checks else [])
    header = f"Evidence this turn ({len(evidence)} tool results):" if evidence else "No tool results this turn."
    used = len(header) + sum(len(line) + 1 for line in tail) + 60  # 60: the omitted-count line
    ranked = sorted(evidence, key=lambda e: (e.status == "ok", _KIND_PRIORITY.get(e.kind, 1), -e.index))
    chosen: list[tuple[int, str]] = []
    for item in ranked[:_DIGEST_MAX_ITEMS]:
        line = f"- {item.evidence_id} {item.tool_name} {item.kind} {item.status}: {_clip(item.target, 70)}"
        if used + len(line) + 1 > DIGEST_BUDGET:
            break
        chosen.append((item.index, line))
        used += len(line) + 1
    omitted = len(evidence) - len(chosen)
    lines = [header, *(line for _, line in sorted(chosen))]
    lines += [f"- … {omitted} more results omitted"] if omitted else []
    return "\n".join([*lines, *tail])[:DIGEST_BUDGET]


def build_evidence_report(answer: str, messages: Sequence[Any], turn_start: int | None = None) -> EvidenceReport:
    """Classify the turn's tool results and check the answer's state claims against them."""
    evidence = classify_turn_evidence(messages, turn_start)
    checks = check_claims(answer, evidence)
    return EvidenceReport(evidence=evidence, claims=checks, digest=_digest(evidence, checks))
