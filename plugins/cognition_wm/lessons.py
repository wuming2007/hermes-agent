"""What the owner's words teach: a standing wish, a correction, a confirmation or a withdrawal.

Deterministic and conservative, like the control intents: a lesson the reader misses can be said
again, a wrong one would steer later work. Only words with something to do in them teach
("以後跑測試都要用 uv run pytest", not "你每次都搞錯"), a request or a question never does, and
words that would drop a check or an approval ("以後推 repo 不用問") are marked so they are never
learned: permissions stay with the task's authorization rules. A user's correction says how they
want the work done; it does not make a technical claim true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from plugins.cognition_wm.model import BEHAVIOR_MAX, EXCEPTIONS_MAX, clip

_FUTURE = r"(?:以後|往後|今後|日後|下次|下一次|每次|每一次|從現在起|從今以後|從現在開始)"
_FUTURE_VERB = r"(?:都|也|請|要|就|記得|務必|一律|必須|先|不要|別|改用|直接|記住|不用|不必|可以)"
# Checked in order: "對，以後都這樣" confirms before it states, "不要再照之前那樣" withdraws.
_CUES: tuple[tuple[str, re.Pattern], ...] = (
    ("revoke", re.compile(
        r"(?:那條|這條|那個|這個|之前的|上次的|剛才的)(?:經驗|規則|教訓|做法|偏好|習慣)(?:是)?(?:不對|錯的|錯了|不適用|作廢|不要了|取消|撤回)"
        r"|(?:忘掉|忘記|刪掉|撤回|取消)(?:那條|這條|那個|這個|之前的|剛才的)?(?:經驗|規則|教訓|做法|偏好)"
        r"|(?:不要|別)再照(?:之前|上次|剛才|那樣|那個)"
        r"|\bforget\s+(?:that|this|the)\s+(?:rule|lesson|preference|habit|approach)\b"
        r"|\b(?:that|this|the)\s+(?:rule|lesson|preference)\s+(?:is|was)\s+wrong\b"
        r"|\bstop\s+(?:doing|following)\s+that\b", re.IGNORECASE)),
    ("result_confirmation", re.compile(
        r"(?:確認|確實|真的|看過|測過|試過|檢查過|驗證過|跑過)[^，。,.!?！？]{0,8}?"
        r"(?:過了|通過|成功|可以用|正常|沒問題|有在跑|上線了|部署好|好了)"
        r"|\b(?:confirmed|verified|checked|tested)\b[^.!?]{0,30}\b(?:works|working|passes|passed|deployed|live|fine)\b"
        r"|\bit\s+(?:really\s+)?works\b|\btests?\s+(?:do\s+)?pass(?:ed)?\s+(?:on\s+my\s+(?:side|machine)|here)\b",
        re.IGNORECASE)),
    ("confirmation", re.compile(
        rf"(?:對|沒錯|是的|正確|很好)[，,。!！\s]*{_FUTURE}(?:都|也|就)*(?:照這樣|這樣做|這樣|照做|這麼做)"
        rf"|{_FUTURE}(?:都|也|就)*(?:照這樣|照做|這麼做)|{_FUTURE}(?:都|也|就)+這樣(?:做|吧|就好)?(?=$|[，。,.!！~])"
        r"|就(?:照|按)這樣|保持這樣"
        r"|\bkeep\s+doing\s+(?:that|this|it)\b"
        r"|\b(?:yes|right|exactly|correct)\b[^.!?]{0,20}\b(?:do\s+it\s+(?:that|this)\s+way|like\s+that)\b",
        re.IGNORECASE)),
    ("directive", re.compile(
        rf"{_FUTURE}[^。！？!?\n]{{0,12}}?{_FUTURE_VERB}|一律(?:用|先|不要|別|要)|永遠(?:不要|別|要|都要)"
        r"|\b(?:from\s+now\s+on|going\s+forward|in\s+(?:the\s+)?future|next\s+time|every\s+time|from\s+here\s+on)\b"
        r"|\b(?:always|never)\s+(?:use|run|do|ask|check|call|push|deploy|commit|write|add|put|include|test|make"
        r"|mention|start|open|close|edit|touch|send)\b", re.IGNORECASE)),
    ("correction", re.compile(
        r"(?<!對)不對(?![？?嗎])|(?<!沒)錯了|不是這樣|不是要|應該(?:要)?(?:用|是|先|改)|改用|換成|要改成|不要用|別用|不該用|用錯"
        r"|\b(?:that'?s|that\s+is|this\s+is)\s+(?:wrong|not\s+right|incorrect)\b|\binstead\s+of\b"
        r"|\buse\s+\S+\s+instead\b|\bshould\s+(?:have\s+)?(?:used?|run|ran|been)\b|\bdon'?t\s+use\b"
        r"|\bwrong\s+(?:one|file|command|approach|branch|database)\b", re.IGNORECASE)),
)
# A directive teaches only when it names something to do; a correction only when it says what to do
# instead ("不對，測試根本沒過" says it is wrong, not how to do it).
_ACTIONS = re.compile(
    r"用|跑|執行|先|改|加|寫|問|確認|檢查|測|部署|推|提交|使用|不要|別|記得|附|標|引用|回覆|回答|格式|命名|放在|存到|換成|列出|說明"
    r"|繁體|中文|英文|直接|跳過|`[^`]+`"
    r"|\b(?:used?|run|check|ask|test|deploy|push|commit|write|add|put|include|avoid|prefer|call|name|format|mention"
    r"|cite|reply|answer|keep|start|stop|merge)\b", re.IGNORECASE)
_PRESCRIBES = re.compile(
    r"要|改用|換成|應該|先|請|不要|別|記得|務必|必須|一律|改成|使用|用(?!錯)|`[^`]+`"
    r"|\b(?:used?|run|should|must|always|never|don'?t|avoid|prefer|instead)\b", re.IGNORECASE)
_TEACHES = {"directive": _ACTIONS, "correction": _PRESCRIBES}
_QUESTION = re.compile(r"是否|有沒有|嗎|[？?]|請|幫我|幫忙|看看|一下|\b(?:can|could|would)\s+you\b|\bplease\b",
                       re.IGNORECASE)
_NEGATIVE_RESULT = re.compile(
    r"沒過|沒有過|沒有通過|沒通過|失敗|不行|沒成功|壞了|不能用|沒(?:有)?(?:部署|上線|生效|更新|改到|跑|完成|修好)"
    r"|\bfail(?:s|ed|ing)?\b|\bbroken\b|\bdoesn'?t\s+work\b|\bnot\s+working\b|\bnever\s+(?:ran|deployed)\b",
    re.IGNORECASE)
# Words that would take away a check or an approval: a permission, never a lesson.
_LOOSENS = re.compile(
    r"不用(?:再)?(?:問|確認|檢查|測試|跑測試)|不必(?:再)?(?:問|確認)|不需要(?:再)?(?:問|確認)|免問|不用經過我"
    r"|(?<!不要)(?<!別)(?<!不能)(?<!不可以)直接(?:推|部署|上線|改|寄|發|提交|執行|合併|merge|push|deploy)"
    r"|跳過(?:測試|檢查|確認|審查)"
    r"|\bno\s+need\s+to\s+(?:ask|check|confirm|test)\b|\bdon'?t\s+(?:bother\s+)?(?:ask|check|confirm)(?:ing)?\b"
    r"|\bwithout\s+asking\b|\byou\s+can\s+just\s+(?:push|deploy|merge|commit|send)\b"
    r"|(?<!n't )(?<!not )\bjust\s+(?:push|deploy|merge|commit|send)\b|\bskip\s+(?:the\s+)?(?:tests?|checks?|review)\b",
    re.IGNORECASE)
_EVERYWHERE = re.compile(
    r"所有(?:的)?專案|每個專案|全部(?:的)?專案|任何專案|不管(?:是)?哪個專案|所有地方"
    r"|\beverywhere\b|\bin\s+(?:all|every|any)\s+(?:projects?|repos?(?:itories)?)\b|\bglobally\b", re.IGNORECASE)
_WRONG = re.compile(
    r"(?<!對)不對(?![？?嗎])|(?<!沒)錯了|搞錯|弄錯|用錯|不是這樣|不是這個|\b(?:wrong|incorrect)\b|\bnot\s+right\b"
    r"|\bthat'?s\s+not\s+(?:it|what\s+i)\b", re.IGNORECASE)
_EXCEPTIONS = re.compile(r"(?:除了|除非|例外(?:是)?|\bexcept(?:\s+for|\s+when)?\b|\bunless\b)[^。.;；!！?？\n]*",
                         re.IGNORECASE)
# What a lesson asks for, first match wins.
_KINDS: tuple[tuple[str, re.Pattern], ...] = (
    ("ask_first", re.compile(r"先問(?:過)?我|問過我|經過我(?:的)?同意|等我(?:確認|同意)|\bask\s+(?:me\s+)?(?:first|before)\b"
                             r"|\bcheck\s+with\s+me\b", re.IGNORECASE)),
    ("procedure", re.compile(r"`[^`]+`|先.{1,20}再|步驟|流程|\b(?:run|use)\s+[\w./-]+"
                             r"|\b[a-z][\w.-]*\s+(?:run|test|install|build|format|lint)\b", re.IGNORECASE)),
    ("caution", re.compile(r"不要|別|不可以|不能|禁止|避免|\b(?:never|don'?t|do\s+not|avoid)\b", re.IGNORECASE)),
)

# The kind of work a request is when no task says so, first match wins: what is being made decides
# ("寫週報，順便列出測試進度" is writing), then what was just done. None when unsure.
_WORK_KINDS: tuple[tuple[str, re.Pattern], ...] = (
    ("writing", re.compile(
        r"寫[^，。,.]{0,4}?(?:週報|日報|月報|報告|信|郵件|文章|文件|草稿|摘要|簡報|說明|講稿|提案|貼文)|草擬|潤稿|改寫|翻譯"
        r"|\b(?:write|draft|rewrite|translate|summari[sz]e)\b[^.!?]{0,20}\b(?:report|email|letter|post|article|doc|summary"
        r"|essay|memo|note)s?\b", re.IGNORECASE)),
    ("ops", re.compile(r"部署|上線|重啟|伺服器|服務狀態|監控|備份|\b(?:deploy|restart|server|rollback|backup)\b",
                       re.IGNORECASE)),
    ("engineering", re.compile(r"測試|程式|bug|編譯|重構|\b(?:pytest|tests?|code|bug|build|refactor|compile)\b",
                               re.IGNORECASE)),
    ("research", re.compile(r"研究|調查|查一下|找資料|比較|差別|\b(?:research|investigate|compare|look\s+up)\b",
                            re.IGNORECASE)),
)
_STEP_WORK = {"terminal": "engineering", "execute_code": "engineering", "patch": "engineering",
              "write_file": "engineering", "process_manage": "engineering"}

_ASCII_WORD = re.compile(r"[A-Za-z][\w.+-]*")
_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_ASCII_STOP = frozenset("""
the and for with from that this these those use used using run runs ran not but you your please always never next
time now instead should dont don't do does did just then when all every any here on in of to is are be it its them
they me my we our
""".split())
_CJK_STOP = frozenset("""
以後 往後 今後 日後 每次 一律 永遠 下次 記得 不要 不用 不必 這個 那個 這樣 那樣 我們 你們 一下 可以 需要 因為 所以 然後 還是 就是
已經 應該 請用 改用 要用 都要 也要 不是 不對 錯了 時候 什麼 怎麼 如果 但是 而且 或是 直接 一定 務必 請你 幫我 你要 我要
""".split())
TAGS_MAX = 24


@dataclass(frozen=True)
class LessonCue:
    kind: str  # directive | correction | confirmation | result_confirmation | revoke
    text: str  # the owner's words, whitespace-normalized and clipped
    everywhere: bool = False  # the user said it holds in every project
    loosens: bool = False  # it would drop a check or an approval: never learned


@dataclass(frozen=True)
class LessonDraft:
    kind: str
    scope: str  # project | global
    situation: str
    behavior: str
    exceptions: str
    tags: tuple[str, ...]
    lesson_key: str


def read_cue(text: str) -> Optional[LessonCue]:
    """What the owner's message teaches, or None."""
    flat = " ".join(str(text or "").split())
    for kind, pattern in _CUES:
        if not pattern.search(flat):
            continue
        if kind == "result_confirmation" and (_QUESTION.search(flat) or _NEGATIVE_RESULT.search(flat)):
            continue
        if kind in _TEACHES and not _TEACHES[kind].search(flat):
            continue
        return LessonCue(kind, clip(flat, 400), everywhere=bool(_EVERYWHERE.search(flat)),
                         loosens=bool(_LOOSENS.search(flat)))
    return None


def work_kind(text: str, steps: Sequence[str] = ()) -> Optional[str]:
    """The kind of work a request is (writing, ops, engineering, research), read from its words or
    else from the tools just used; None when unsure, so an unclear request is never held back."""
    flat = " ".join(str(text or "").split())
    found = next((kind for kind, pattern in _WORK_KINDS if pattern.search(flat)), None)
    return found or next((_STEP_WORK[s.split(":", 1)[0]] for s in steps if s.split(":", 1)[0] in _STEP_WORK), None)


def says_wrong(text: str) -> bool:
    """Whether the owner says what was just done or claimed is wrong (a correction to count)."""
    flat = " ".join(str(text or "").split())
    return bool(_WRONG.search(flat) or _NEGATIVE_RESULT.search(flat))


def correction_kind(cue: Optional[LessonCue], text: str) -> Optional[str]:
    """What the owner corrects in the last answer: ``result`` (it did not hold — "測試根本沒過",
    "不對" with nothing else to do), ``approach`` (how it was done — "要用 uv run pytest"), or None.
    Correcting the approach says nothing about whether a result our own check showed was true."""
    flat = " ".join(str(text or "").split())
    if _NEGATIVE_RESULT.search(flat):
        return "result"
    if cue is not None and cue.kind == "correction":
        return "approach"
    return "result" if _WRONG.search(flat) else None


def _ascii_terms(text: str) -> list[str]:
    words = (word.lower().strip(".-+") for word in _ASCII_WORD.findall(text or ""))
    return [word for word in words if len(word) >= 2 and word not in _ASCII_STOP]


def _bigrams(text: str, *, drop_common: bool = True) -> list[str]:
    grams = [run[i:i + 2] for run in _CJK_RUN.findall(text or "") for i in range(len(run) - 1)]
    return [gram for gram in grams if not drop_common or gram not in _CJK_STOP]


def search_terms(text: str) -> list[str]:
    """Words a lesson can be found by: English words, and CJK text by its two-character pieces
    (Chinese has no spaces, so a whole phrase would only ever match itself)."""
    return list(dict.fromkeys([*_ascii_terms(text), *_bigrams(text)]))


def topic_terms(text: str) -> frozenset:
    """What a lesson is about: its command and tool words when it names two or more, else its words."""
    ascii_terms = list(dict.fromkeys(_ascii_terms(text)))
    return frozenset(ascii_terms if len(ascii_terms) >= 2 else [*ascii_terms, *_bigrams(text)])


def same_topic(a: frozenset, b: frozenset) -> bool:
    """Two lessons about the same thing; one shared word is not a topic."""
    shared = a & b
    return len(shared) >= 2 and len(shared) / min(len(a), len(b)) >= 0.6


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower())


def same_wording(a: str, b: str) -> bool:
    """The same lesson said again (punctuation and spacing aside), not a revised one."""
    if _normalized(a) == _normalized(b):
        return True
    terms_a = set(_ascii_terms(a) + _bigrams(a, drop_common=False))
    terms_b = set(_ascii_terms(b) + _bigrams(b, drop_common=False))
    return bool(terms_a and terms_b) and len(terms_a & terms_b) / len(terms_a | terms_b) >= 0.8


def draft_lesson(cue: LessonCue, *, focus_title: str, focus_kind: Optional[str], previous_steps: Sequence[str],
                 previous_request: str = "") -> LessonDraft:
    """The lesson a directive or correction states: the user's own words as what to do, the
    exception they named, where it applies (the task, else what the user asked and what was done),
    and the words it can be found by — the request's too, since a correction rarely repeats it."""
    found = _EXCEPTIONS.search(cue.text)
    exceptions = clip(found.group(0).strip(), EXCEPTIONS_MAX) if found else ""
    behavior = cue.text if not found else cue.text[:found.start()] + cue.text[found.end():]
    behavior = clip(behavior.strip(" ，,。;；"), BEHAVIOR_MAX)
    if focus_title and focus_kind:
        situation = f'{focus_kind} task "{clip(focus_title, 60)}"'
    elif previous_request:
        situation = f'asked "{clip(previous_request, 60)}"' + (f", after {previous_steps[0]}" if previous_steps else "")
    elif previous_steps:
        situation = "after " + "; ".join(previous_steps[:2])
    else:
        situation = "this project"
    kind = next((name for name, pattern in _KINDS if pattern.search(behavior)), "preference")
    tags = tuple(dict.fromkeys([*search_terms(behavior), *search_terms(focus_title),
                                *search_terms(previous_request)]))[:TAGS_MAX]
    return LessonDraft(kind=kind, scope="global" if cue.everywhere else "project", situation=situation,
                       behavior=behavior, exceptions=exceptions, tags=tags,
                       lesson_key=" ".join(sorted(topic_terms(behavior))))
