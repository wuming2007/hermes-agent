"""Behavior contract: which sentences of an answer claim a state that evidence must back.

A false claim here makes the evidence gate nag an honest answer; a missed one lets an unverified
"done" through. Expectations are hand-written literals.
"""

import pytest

from agent.cognition_claims import state_claims


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("I pushed back the deadline by a week.", id="idiom"),
        pytest.param("The race condition created this bug.", id="explaining-a-cause"),
        pytest.param("An updated dependency broke the build.", id="participle-as-adjective"),
        pytest.param("Modified state caused the crash.", id="sentence-initial-adjective"),
        pytest.param("The app restarted itself after an OOM kill.", id="something-else-acted"),
        pytest.param("Let me know when you're done.", id="the-user-is-done"),
        pytest.param("Make sure the tests pass before merging.", id="advice"),
        pytest.param("I have not deployed it yet.", id="english-negation"),
        pytest.param("部署完成後，服務會自動重啟。", id="condition-after"),
        pytest.param("如果測試通過，就可以合併。", id="condition-before"),
        pytest.param("需要的話，我可以幫你部署。", id="offer"),
        pytest.param("要我現在部署嗎？", id="question"),
        pytest.param("目前尚未部署，也還沒跑測試。", id="cjk-negation"),
        pytest.param("```\necho done\ngit push\n```", id="code-block"),
        pytest.param("今天天氣不錯，適合出去走走。", id="chat"),
    ],
)
def test_explanations_conditions_offers_and_code_are_not_claims(answer):
    assert state_claims(answer) == ()


@pytest.mark.parametrize(
    "answer, classes",
    [
        pytest.param("I've deployed the new build.", ["deployed"], id="first-person"),
        pytest.param("Updated config.yaml and restarted the gateway.", ["deployed", "written"], id="report-line"),
        pytest.param("- Created tests/test_app.py", ["written"], id="bullet"),
        pytest.param("The bug is fixed and all tests pass.", ["tests_pass", "working"], id="two-claims-english"),
        pytest.param("已重新啟動服務，設定檔也已寫入完成。", ["deployed", "written"], id="two-claims-one-sentence"),
        pytest.param("已部署完成。", ["deployed"], id="shared-words-counted-once"),
        pytest.param("我已經幫你寄出郵件了。", ["completed"], id="words-between-already-and-verb"),
        pytest.param("我已寫入設定檔。測試全部通過。", ["written", "tests_pass"], id="sentences-without-spaces"),
    ],
)
def test_state_claims_are_found_per_sentence_and_class(answer, classes):
    assert [c.claim_class for c in state_claims(answer)] == classes


def test_attribution_covers_following_sentences_until_we_speak_for_ourselves():
    claims = state_claims("Claude 回報已完成修改。測試都通過了。我也已經部署完成。")

    assert [(c.claim_class, c.attributed) for c in claims] == [
        ("written", True), ("tests_pass", True), ("deployed", False)]
