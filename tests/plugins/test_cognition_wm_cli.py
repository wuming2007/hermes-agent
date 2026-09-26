"""Behavior contract: `hermes cognition lessons` shows what was learned for this project and lets
the user settle it — their word on the command line counts as theirs, and forget deletes for good.

The real plugin CLI and store under the test's isolated HERMES_HOME.
"""

import argparse
import os
from pathlib import Path

from plugins.cognition_wm import cli, runtime
from plugins.cognition_wm.model import SourceRef


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes cognition")
    cli.register_cli(parser)
    return parser.parse_args(argv)


def _run(capsys, *argv: str) -> str:
    code = cli.cognition_command(_parse(*argv))
    out = capsys.readouterr().out
    assert code == 0, out
    return out


def _lesson(project: str, status: str = "candidate"):
    proposed = SourceRef(type="assistant", session_id="s1", turn_id="t1", ref="task_state", at=1.0,
                         authored_by="model", run_id="run-a")
    return runtime.store().add_experience(
        kind="procedure", scope="project", project_key=project, task_kind="engineering", situation="running the tests",
        behavior="use `uv run pytest`", tags=("pytest",), lesson_key="pytest uv", status=status, origin=proposed)


def test_the_user_sees_this_projects_lessons_and_where_each_came_from(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    here = _lesson(runtime.project_key())
    elsewhere = _lesson("git:/elsewhere")

    out = _run(capsys, "lessons")
    assert here.experience_id in out and "uv run pytest" in out and "suggested by the assistant" in out
    assert elsewhere.experience_id not in out
    assert elsewhere.experience_id in _run(capsys, "lessons", "--all-projects")


def test_the_users_word_on_the_command_line_settles_a_lesson(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lesson = _lesson(runtime.project_key())

    _run(capsys, "lessons", "confirm", lesson.experience_id)
    confirmed = runtime.store().experience(lesson.experience_id)
    assert (confirmed.status, confirmed.confirmed_at is not None) == ("active", True)
    assert [s.source_class for s in runtime.store().signals(lesson.experience_id)] == ["user_confirmation"]
    _run(capsys, "lessons", "revoke", lesson.experience_id)
    assert runtime.store().experience(lesson.experience_id).status == "revoked"


def test_forget_deletes_for_good(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    one, _two = _lesson(runtime.project_key()), _lesson(runtime.project_key())
    kept = _lesson("git:/elsewhere")

    _run(capsys, "lessons", "forget", one.experience_id)
    assert runtime.store().experience(one.experience_id) is None
    _run(capsys, "lessons", "forget", "--all")
    assert [e.experience_id for e in runtime.store().experiences()] == [kept.experience_id]


def test_a_lesson_of_another_project_or_an_unknown_id_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    elsewhere = _lesson("git:/elsewhere")

    for experience_id in ("xp_000000000000", elsewhere.experience_id):
        assert cli.cognition_command(_parse("lessons", "revoke", experience_id)) == 1
        assert "no such lesson" in capsys.readouterr().out
    assert runtime.store().experience(elsewhere.experience_id).status == "candidate"


def test_the_plugin_puts_the_command_on_the_hermes_cli(tmp_path, monkeypatch):
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [cognition_wm]\n", encoding="utf-8")
    discover_plugins(force=True)

    assert "cognition" in get_plugin_manager()._cli_commands
