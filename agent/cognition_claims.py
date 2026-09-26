"""What an answer claims about the state of the world, sentence by sentence.

Pure and deterministic. Finds the state claims an answer makes — deployed, tests pass, verified,
fixed/working, written, done — and skips what only sounds like one: negated statements,
conditions and advice, offers, questions, code, and English prose that uses the same verbs to
explain something ("the race condition created this bug"). It also notes when a sentence relays
someone else's account (a subagent's report) instead of our own result.
``agent.cognition_evidence`` judges each claim against the turn's tool results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.process_monitor import extract_claims_from_response

MAX_CLAIMS = 16

_CJK_DONE = r"(?:完成|好了|成功|完畢|了)"
_CJK_SINCE = r"已經?[^，,。；;！!？?\n]{0,6}?"  # 已／已經 plus a short object: 已經幫你寄出

# An English state claim needs a reporting frame: a first-person subject ("I deployed"), a
# present-perfect or "is now" state ("has been saved"), or a report line / coordinated clause
# that opens with the verb and an object ("Updated config.yaml and restarted the gateway").
_EN_SELF = r"\b(?:i|we)(?:'ve|'d| have| had)?\s+(?:(?:now|just|already|also|successfully)\s+)*"
_EN_STATE = r"\b(?:has|have)\s+(?:now\s+|already\s+)?been\s+(?:successfully\s+)?|\b(?:is|are)\s+now\s+"
_EN_OBJECT = r"(?::|\s+(?:the|a|an|your|my|our|this|that|these|those|all|both|it|them|to|in|on|\S*[./_]\S*)\b)"


def _en(verbs: str, state: str = _EN_STATE) -> str:
    return (rf"(?:{_EN_SELF}|{state})(?:{verbs})\b"
            rf"|(?:^|(?:\band|\bthen|,)\s+)(?:{verbs}){_EN_OBJECT}")


_EN_DEPLOYED = _en(r"(?:re)?deployed|restarted|released|rolled out|shipped|published|pushed(?!\s+back)")
_EN_VERIFIED = _en("verified|confirmed|double-checked|validated|checked")
_EN_WORKING = _en("fixed|resolved|repaired", state=_EN_STATE + r"|\b(?:is|are)\s+")
_EN_WRITTEN = _en("wrote|written|saved|created|updated|modified|edited|added|changed")

# Precedence order: where two claims share words ("已部署完成"), the earlier class keeps them.
CLAIM_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("deployed", re.compile(
        rf"{_CJK_SINCE}(?:部署|佈署|上線|重啟|重新啟動|發布|發佈|推送|推上)|(?:部署|佈署|上線|重啟|重新啟動|發布|發佈){_CJK_DONE}"
        rf"|{_EN_DEPLOYED}"
        r"|\b(?:is|are)\s+(?:now\s+)?live\b|\bwent live\b", re.IGNORECASE)),
    ("tests_pass", re.compile(
        r"測試[^，,。；;\n]{0,12}?(?:通過|過了|全過|全綠)|\bCI\b[^，,。；;\n]{0,8}?(?:通過|綠)"
        r"|\b(?:tests?|specs?|CI|checks?)\b[^.,;\n]{0,20}?\b(?:pass(?:ed|es|ing)?|green|succeed(?:ed|s)?)\b",
        re.IGNORECASE)),
    ("verified", re.compile(
        r"已經?(?:驗證|核對)|(?:驗證|確認|核對|檢查)(?:過了|過|完成|無誤|正常|通過)"
        rf"|{_EN_VERIFIED}", re.IGNORECASE)),
    ("working", re.compile(
        r"修(?:好|復|復好)|(?:可以|能夠?)正常(?:使用|運作|運行|執行|啟動|工作)|正常運作|運作正常|恢復正常|已經?解決|解決了"
        rf"|{_EN_WORKING}"
        r"|\b(?:it|this|that|everything|the\s+\w+)\s+(?:now\s+)?works\b|\bworks\s+(?:now|again|correctly|fine|as expected)\b"
        r"|\b(?:is|are)\s+(?:now\s+)?working\b(?!\s+on\b)", re.IGNORECASE)),
    ("written", re.compile(
        rf"{_CJK_SINCE}(?:寫入|寫好|建立|新增|更新|修改|改好|儲存|存檔|存好|產生|生成|加入)|完成(?:了)?(?:修改|寫入|建立|新增|更新|編輯)"
        rf"|(?:寫入|建立|新增|儲存|產生|生成){_CJK_DONE}"
        rf"|{_EN_WRITTEN}", re.IGNORECASE)),
    ("completed", re.compile(
        rf"{_CJK_SINCE}(?:完成|搞定|做完|處理好|寄出|發送|送出|寄送|傳送|通知)|(?:全部|都|均)(?:已經?)?(?:完成|搞定|做完)"
        r"|(?:完成|搞定|做完)了"
        r"|\b(?:i|we)(?:'ve| have)?\s+(?:(?:now|just|already)\s+)*(?:finished|completed|sent)\b"
        r"|\b(?:i|we)(?:'m|'re| am| are)\s+(?:all\s+)?done\b|^(?:done|all set|finished|completed)\b|\ball set\b"
        r"|\b(?:it|this|that|everything|all)(?:'s|\s+is|\s+are|\s+has been|\s+have been)\s+(?:now\s+)?"
        r"(?:done|complete|completed|finished|sent)\b", re.IGNORECASE)),
)
CLAIM_CLASSES = tuple(name for name, _ in CLAIM_PATTERNS)

# Words in the same clause, before a match or inside it, that make it no claim at all.
_NEGATION = re.compile(
    r"尚未|還沒|還未|沒有|並未|未能|無法|不能|不會|不是|沒|未|不"
    r"|\b(?:not|never|no|yet to|unable|cannot|can't|without)\b|n't\b", re.IGNORECASE)
_HYPOTHETICAL = re.compile(
    r"是否|等待|待確認|如果|若|假如|一旦|確保|需要|應該|必須|直到|要讓"
    r"|\b(?:if|when|once|until|unless|whether|make sure|ensure|should|must|needs? to|so that)\b", re.IGNORECASE)
_AFTER_CONDITION = re.compile(r"\s*(?:後|之後|以後|的話)")  # 部署完成後… is a condition
_CLAUSE_BREAK = re.compile(r"[，,。；;！!？?：:\n]")
_WINDOW = 16

_ATTRIBUTION = re.compile(r"回報|表示|聲稱|指出|據|說|\b(?:reports?|reported|says|said|claims?|according to)\b",
                          re.IGNORECASE)
_FIRST_PERSON = re.compile(r"我|\b(?:i|we)\b", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_LEADING_MARKS = re.compile(r"^[^\w`]+")


@dataclass(frozen=True)
class StateClaim:
    sentence: str
    claim_class: str
    attributed: bool  # the sentence relays someone else's account (e.g. "Claude 回報…")


def _sentences(answer: str) -> list[str]:
    text = _FENCED_CODE.sub(" ", answer or "").replace("**", "")
    claims = extract_claims_from_response(text, max_claims=4 * MAX_CLAIMS)
    return [s for s in (_LEADING_MARKS.sub("", claim.text).strip() for claim in claims) if s]


def _asserted(sentence: str, match: re.Match) -> bool:
    breaks = [m.end() for m in _CLAUSE_BREAK.finditer(sentence, 0, match.start())]
    window = sentence[max(breaks[-1] if breaks else 0, match.start() - _WINDOW):match.end()]
    return not (_NEGATION.search(window) or _HYPOTHETICAL.search(window)
                or _AFTER_CONDITION.match(sentence, match.end()))


def _claim_span(pattern: re.Pattern, sentence: str, taken: list[tuple[int, int]]) -> tuple[int, int] | None:
    for match in pattern.finditer(sentence):
        overlaps = any(match.start() < end and start < match.end() for start, end in taken)
        if not overlaps and _asserted(sentence, match):
            return match.span()
    return None


def state_claims(answer: str) -> tuple[StateClaim, ...]:
    """Every state claim in the answer, in sentence order and class precedence within a sentence.

    An attribution ("Claude 回報…", "the subagent reports…") covers its sentence and the ones
    after it until a sentence speaks in the first person again."""
    claims: list[StateClaim] = []
    attributed = False
    for sentence in _sentences(answer):
        if _ATTRIBUTION.search(sentence):
            attributed = True
        elif _FIRST_PERSON.search(sentence):
            attributed = False
        if sentence.endswith(("?", "？")):
            continue
        taken: list[tuple[int, int]] = []
        for claim_class, pattern in CLAIM_PATTERNS:
            span = _claim_span(pattern, sentence, taken)
            if span is not None:
                taken.append(span)
                claims.append(StateClaim(sentence, claim_class, attributed))
    return tuple(claims[:MAX_CLAIMS])
