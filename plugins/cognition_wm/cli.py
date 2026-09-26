"""``hermes cognition``: see and settle what Hermes learned (cognition v2, phase 2).

    hermes cognition lessons [--all-projects] [--status STATUS]   the lessons of this project
    hermes cognition lessons show ID                              one lesson and what it rests on
    hermes cognition lessons confirm ID | revoke ID               your own word on a lesson
    hermes cognition lessons forget ID | --all [--all-projects]   delete for good

Runs outside any turn: the project is the current directory's, the profile the active one. Your
word here counts as yours, the same as saying it in a chat. No model tool is involved.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Optional

from plugins.cognition_wm import runtime
from plugins.cognition_wm.learning import support_label
from plugins.cognition_wm.model import EXPERIENCE_STATUSES, Experience, SourceRef
from plugins.cognition_wm.provenance import RUN_ID
from plugins.cognition_wm.store import WorkingMemoryError


def register_cli(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="cognition_command")
    lessons = subs.add_parser("lessons", help="List, show, confirm, revoke or forget what Hermes learned")
    lessons.add_argument("op", nargs="?", default="list", choices=("list", "show", "confirm", "revoke", "forget"))
    lessons.add_argument("experience_id", nargs="?")
    lessons.add_argument("--all-projects", action="store_true", help="every project, not just this one")
    lessons.add_argument("--status", choices=EXPERIENCE_STATUSES, help="only lessons in this state")
    lessons.add_argument("--all", action="store_true", help="with forget: every lesson (of this project)")


def _yours() -> SourceRef:
    now = time.time()
    return SourceRef(type="user_message", session_id="cli", turn_id=f"cli-{now:.6f}", ref="hermes cognition", at=now,
                     authored_by="owner", run_id=RUN_ID)


def _in_reach(lesson: Optional[Experience], args: argparse.Namespace) -> bool:
    return lesson is not None and (args.all_projects or lesson.scope == "global"
                                   or lesson.project_key == runtime.project_key())


def _when(stamp: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(stamp)) if stamp else "—"


def _describe(lesson: Experience, *, full: bool) -> str:
    wm = runtime.store()
    where = ("every project" if lesson.scope == "global"
             else "this project" if lesson.project_key == runtime.project_key() else lesson.project_key)
    head = (f"{lesson.experience_id}  {lesson.status:<10} {lesson.kind:<10} {where} · "
            f"{lesson.task_kind or 'any work'} · since {_when(lesson.created_at)}")
    lines = [head, f"  when:    {lesson.situation}", f"  do:      {lesson.behavior}"]
    if lesson.exceptions:
        lines.append(f"  except:  {lesson.exceptions}")
    lines.append(f"  source:  {support_label(wm, lesson)}")
    if full:
        env = lesson.learned_under
        lines.append(f"  learned: {env.get('model') or '?'} / {env.get('provider') or '?'} / "
                     f"Hermes {env.get('hermes_version') or '?'}")
        if lesson.supersedes:
            lines.append(f"  replaces: {lesson.supersedes}")
        for signal in wm.signals(lesson.experience_id):
            lines.append(f"  {'+' if signal.polarity > 0 else '-'} {signal.source_class:<22} {_when(signal.at)}  "
                         f"{signal.detail}")
    return "\n".join(lines)


def _list(args: argparse.Namespace) -> int:
    wm = runtime.store()
    statuses = (args.status,) if args.status else ("candidate", "active", "stale")
    found = wm.experiences(project_key=None if args.all_projects else runtime.project_key(), statuses=statuses)
    if not found:
        print("No lessons kept here yet.")
        return 0
    print("\n\n".join(_describe(lesson, full=False) for lesson in found))
    return 0


def _one(args: argparse.Namespace) -> Optional[Experience]:
    lesson = runtime.store().experience(args.experience_id or "")
    if not _in_reach(lesson, args):
        print(f"no such lesson here: {args.experience_id or '(no id given)'}")
        return None
    return lesson


def _show(args: argparse.Namespace) -> int:
    lesson = _one(args)
    if lesson is None:
        return 1
    print(_describe(lesson, full=True))
    return 0


def _settle(args: argparse.Namespace, status: str) -> int:
    lesson = _one(args)
    if lesson is None:
        return 1
    wm, yours = runtime.store(), _yours()
    try:
        if status == "active":
            wm.add_signal(lesson.experience_id, source_class="user_confirmation", polarity=1, source=yours,
                          evidence_key=f"cli:{yours.turn_id}", lineage="cli")
        lesson = wm.set_experience_status(lesson.experience_id, status, source=yours)
    except WorkingMemoryError as exc:
        print(f"not changed: {exc}")
        return 1
    print(f"{lesson.experience_id} is now {lesson.status}.")
    return 0


def _forget(args: argparse.Namespace) -> int:
    wm = runtime.store()
    if args.all:
        count = wm.forget_experiences(project_key=None if args.all_projects else runtime.project_key())
        print(f"Forgot {count} lesson(s).")
        return 0
    lesson = _one(args)
    if lesson is None:
        return 1
    wm.forget_experience(lesson.experience_id)
    print(f"Forgot {lesson.experience_id}.")
    return 0


# lessons op -> handler
_LESSON_OPS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": _list, "show": _show, "confirm": lambda a: _settle(a, "active"), "revoke": lambda a: _settle(a, "revoked"),
    "forget": _forget,
}


def _lessons(args: argparse.Namespace) -> int:
    return _LESSON_OPS[args.op](args)


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {"lessons": _lessons}


def cognition_command(args: argparse.Namespace) -> int:
    handler = _COMMANDS.get(getattr(args, "cognition_command", None) or "")
    if handler is None:
        print("usage: hermes cognition lessons [list|show|confirm|revoke|forget] [ID]")
        return 1
    return handler(args)
