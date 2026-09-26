"""Behavior contract: which of the user's words become a lesson, and what a lesson keeps.

Pure and deterministic; the examples are hand-written. Conservative on purpose: a missed lesson can
be stated again, a wrong one would steer later work.
"""

import pytest

from plugins.cognition_wm.lessons import draft_lesson, read_cue, same_topic, same_wording, says_wrong, topic_terms


@pytest.mark.parametrize("text, kind", [
    # a standing wish, stated for the future
    ("以後跑測試都要用 uv run pytest", "directive"),
    ("每次改完都先跑 lint 再給我看", "directive"),
    ("下次記得先確認分支", "directive"),
    ("From now on, run the tests with `uv run pytest`.", "directive"),
    ("Always use ruff format before committing.", "directive"),
    ("以後部署前都先問我", "directive"),
    # a correction of what was just done
    ("不對，要用 uv run pytest，不是直接 pytest", "correction"),
    ("不要用 pytest，改用 uv run pytest", "correction"),
    ("That's wrong, use pnpm instead of npm here.", "correction"),
    ("You should have used the staging database.", "correction"),
    # the user confirms a lesson, or a result
    ("對，以後都這樣做", "confirmation"),
    ("就照這樣", "confirmation"),
    ("好，以後就這樣", "confirmation"),
    ("Yes, keep doing that.", "confirmation"),
    ("我確認過了，測試真的通過", "result_confirmation"),
    ("I checked, it works on my machine.", "result_confirmation"),
    # the user withdraws a lesson
    ("那條規則不對，忘掉它", "revoke"),
    ("不要再照之前那樣做", "revoke"),
    ("Forget that rule, it was wrong.", "revoke"),
])
def test_the_users_words_are_read_for_what_they_teach(text, kind):
    cue = read_cue(text)
    assert cue is not None and cue.kind == kind


@pytest.mark.parametrize("text", [
    "幫我跑一下測試",                   # a request, not a lesson
    "今天天氣不錯",                     # chat
    "對不對？",                         # a question
    "你每次都搞錯",                     # a complaint that says nothing to do
    "確認一下測試有沒有通過",             # asks for a check; confirms nothing
    "只要改這一行",                     # scope of this task, not a standing wish
    "OK",
    "不對，測試根本沒過",                 # says it is wrong, not how to do it
])
def test_requests_questions_and_complaints_teach_nothing(text):
    assert read_cue(text) is None


@pytest.mark.parametrize("text, wrong", [
    ("不對，測試根本沒過", True),
    ("你每次都搞錯", True),
    ("That is not right.", True),
    ("對不對？", False),
    ("測試都過了嗎", False),
])
def test_saying_the_last_answer_was_wrong_is_read_on_its_own(text, wrong):
    assert says_wrong(text) is wrong


@pytest.mark.parametrize("text", [
    "以後推 repo 不用問我",
    "以後可以直接部署",
    "From now on you can just push to main.",
    "下次跳過測試直接交",
])
def test_words_that_would_drop_a_check_or_an_approval_are_never_learned(text):
    cue = read_cue(text)
    assert cue is not None and cue.loosens


@pytest.mark.parametrize("text, everywhere", [
    ("以後所有專案都用 uv run pytest", True),
    ("Always use conventional commits in every repo.", True),
    ("以後這個專案都用 uv run pytest", False),
])
def test_a_lesson_holds_for_every_project_only_when_the_user_says_so(text, everywhere):
    assert read_cue(text).everywhere is everywhere


def test_a_lesson_keeps_the_users_words_its_exceptions_and_searchable_terms():
    cue = read_cue("以後跑測試都要用 uv run pytest，除了 docs 目錄以外")
    lesson = draft_lesson(cue, focus_title="跑單元測試", focus_kind="engineering", previous_steps=())

    assert "uv run pytest" in lesson.behavior
    assert lesson.exceptions.startswith("除了")
    assert {"測試", "pytest", "uv"} <= set(lesson.tags)  # a CJK word is findable by its own characters
    assert lesson.situation == 'engineering task "跑單元測試"'
    assert lesson.kind == "procedure"


@pytest.mark.parametrize("text, kind", [
    ("以後部署前都先問我", "ask_first"),
    ("以後不要直接推 main", "caution"),
    ("以後跑測試都要用 `uv run pytest`", "procedure"),
    ("以後回覆都用繁體中文", "preference"),
])
def test_a_lesson_is_filed_by_what_it_asks_for(text, kind):
    assert draft_lesson(read_cue(text), focus_title="", focus_kind=None, previous_steps=()).kind == kind


def test_without_a_task_the_situation_is_what_was_just_done():
    lesson = draft_lesson(read_cue("不對，改用 uv run pytest"), focus_title="", focus_kind=None,
                          previous_steps=("terminal: pytest -q",))
    assert lesson.situation == "after terminal: pytest -q"


def test_two_wordings_of_one_lesson_are_one_topic_and_other_lessons_are_not():
    a = topic_terms("以後跑測試都要用 uv run pytest")
    b = topic_terms("不要直接跑 pytest，要用 uv run pytest")
    c = topic_terms("以後部署前都先問我")

    assert same_topic(a, b)
    assert not same_topic(a, c)
    assert not same_topic(topic_terms("用 pytest"), topic_terms("部署前先跑 pytest 再推"))  # one shared word is not a topic


def test_a_repeated_lesson_reads_the_same_and_a_revised_one_does_not():
    assert same_wording("以後跑測試都要用 uv run pytest", "以後跑測試，都要用 uv run pytest！")
    assert not same_wording("以後跑測試都要用 uv run pytest", "不要用 uv 了，直接跑 pytest")


@pytest.mark.parametrize("text, steps, kind", [
    ("幫我寫這週的週報", (), "writing"),
    ("幫我寫週報，順便列出測試相關的進度", (), "writing"),  # what is being made decides, not a word in it
    ("Draft an email to the team", (), "writing"),
    ("跑一下測試", (), "engineering"),
    ("幫我修這個 bug", (), "engineering"),
    ("部署到正式機", (), "ops"),
    ("幫我查一下這兩個方案的差別", (), "research"),
    ("繼續", ("terminal: pytest -q",), "engineering"),  # nothing in the words: what was just done
    ("好", (), None),  # unsure says nothing
])
def test_the_kind_of_work_is_read_from_what_is_asked_when_no_task_says(text, steps, kind):
    from plugins.cognition_wm.lessons import work_kind

    assert work_kind(text, steps) == kind
