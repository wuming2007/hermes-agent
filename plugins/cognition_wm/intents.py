"""Control intents in the owner's words: stop, pause, resume, or limit the work.

Conservative on purpose. A hit only marks the intent as pending: the snapshot asks the model to
record it with ``task_state``, and until it does the scope gate sends execution tools to the
human-approval prompt. A limit is recognized only when it names what not to do ("不要改 repo",
"只要草稿", "don't touch the code"), so "只要改這一行" is not one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from plugins.cognition_wm.model import clip


@dataclass(frozen=True)
class ControlIntent:
    kind: str  # cancel | pause | restrict | resume
    text: str  # the owner's words, clipped


_ACTIONS_ZH = r"(?:改|動|碰|修改|寄|寄出|發|發送|送出|部署|上線|推|提交|執行|跑)"
_ACTIONS_EN = r"(?:change|modify|touch|edit|send|deploy|push|commit|run|execute)"
_OUTPUTS_ZH = r"(?:prompt|提示詞|提示|草稿|計畫|大綱|說明|建議|方案)"

# Checked in order: "不要做了" is a cancel, not a limit.
_INTENTS: tuple[tuple[str, re.Pattern], ...] = (
    ("cancel", re.compile(
        r"取消(?:這個|這項|任務|工作|吧|了)?|不要做了|不用做了|別做了|算了|放棄(?:這個|吧)?|停掉"
        r"|\b(?:cancel|abort|scrap|drop)\s+(?:it|this|that|the\s+(?:task|work|plan|job))\b"
        r"|\bstop\s+(?:working\s+on|doing)\b|\bnever\s*mind\b|\bforget\s+(?:it|about\s+it)\b", re.IGNORECASE)),
    ("pause", re.compile(
        r"暫停|先停|先不要(?:做|動|改|繼續)|等一下再|晚點再"
        r"|\bpause\b|\bhold\s+(?:off|on)\b|\bput\s+(?:it|this)\s+on\s+hold\b", re.IGNORECASE)),
    ("restrict", re.compile(
        rf"只要[^，。,.；;]{{0,8}}?{_OUTPUTS_ZH}|不要{_ACTIONS_ZH}(?!得)|別{_ACTIONS_ZH}(?!得)"
        rf"|\b(?:only|just)\s+(?:the\s+)?(?:prompt|plan|draft|outline|text|advice)\b"
        rf"|\b(?:don'?t|do\s+not|never)\s+{_ACTIONS_EN}\b|\bwithout\s+(?:changing|touching|sending|deploying)\b",
        re.IGNORECASE)),
    ("resume", re.compile(
        r"繼續(?:做|吧|剛才|之前|上次)|恢復(?:任務|工作)|接著做"
        r"|\b(?:resume|carry\s+on|pick\s+(?:it|this)\s+(?:back\s+)?up)\b", re.IGNORECASE)),
)


def detect_intent(text: str) -> Optional[ControlIntent]:
    """The first control intent in the owner's message, or None."""
    for kind, pattern in _INTENTS:
        if pattern.search(text or ""):
            return ControlIntent(kind, clip(text, 200))
    return None
