"""Behavior contract: control intents are recognized in the owner's words, and ordinary remarks are not.

A false hit asks the model to record a limit and sends execution tools to the approval prompt, so
the lexicon must stay narrow; a miss lets a "stop" slide.
"""

import pytest

from plugins.cognition_wm.intents import detect_intent


@pytest.mark.parametrize(
    "text, kind",
    [
        pytest.param("取消這個任務", "cancel", id="cancel"),
        pytest.param("先不要做了", "cancel", id="stop-doing-it"),
        pytest.param("never mind, drop it", "cancel", id="english-cancel"),
        pytest.param("暫停一下，我想想", "pause", id="pause"),
        pytest.param("put this on hold for now", "pause", id="english-pause"),
        pytest.param("只要幫我寫給 Claude 的 prompt，不要改 repo", "restrict", id="prompt-only"),
        pytest.param("寫週報，只要草稿，不要寄出", "restrict", id="draft-only"),
        pytest.param("don't touch the code, just give me the plan", "restrict", id="english-limit"),
        pytest.param("繼續做剛才的重構", "resume", id="resume"),
    ],
)
def test_control_intents_are_recognized(text, kind):
    intent = detect_intent(text)

    assert intent is not None and intent.kind == kind


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("只要改這一行內容就好", id="a-small-change-is-still-a-change"),
        pytest.param("程式碼不要跑得太慢", id="a-performance-remark"),
        pytest.param("只要改這一行", id="scope-of-a-change"),
        pytest.param("繼續說", id="keep-talking"),
        pytest.param("這個函式要怎麼寫？", id="a-question"),
        pytest.param("The plan should not change the schema.", id="describing-a-plan"),
    ],
)
def test_ordinary_remarks_are_not_control_intents(text):
    assert detect_intent(text) is None
