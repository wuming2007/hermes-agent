"""Cognition v2 evaluation: the same cases in three arms, with the model of an eval home.

``off`` is the existing cognition (routing on, no working memory, no evidence loop); ``on`` is
phase 1 (the cognition_wm plugin and the evidence loop); ``v2`` is phase 1 plus learning and
calibration. A case is one or more conversations (``sessions``, like the same user on another
day), each turn run as a fresh agent (like ``hermes chat -r``) in the case's own directories; each
turn writes one JSONL row (latency, API calls, tokens, tool calls and how many were blocked, the
working-memory and lessons characters added, the evidence-loop outcome, whether the answer check
ran, the case's checks for that turn, the final answer), and each case run ends with a summary row
(what was learned, how the recorded claims held up). Every case run starts with nothing learned.

Runs only in a dedicated eval home made with ``--init-home`` (never the everyday ``~/.hermes``).
The toolsets are file, terminal and task_state: no messaging, web, cron or subagents, so a case
cannot reach anyone. The user logs in to the eval home themselves; nothing copies credentials,
and the eval home never borrows the Codex CLI or Claude Code login (``auth.adopt_external_logins``
is turned off by ``--init-home`` and checked before every run).

    python evals/cognition_v2/runner.py --init-home ~/hermes-eval-home
    HERMES_HOME=~/hermes-eval-home hermes auth add openai-codex --type oauth   # the user, once
    python evals/cognition_v2/runner.py --home ~/hermes-eval-home \\
        --cases evals/cognition_v2/cases.phase2.example.jsonl --out /tmp/cv2.jsonl --set dev --repeats 3
    python evals/cognition_v2/runner.py --summarize /tmp/cv2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # run as a script: import the checkout, not an installed copy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.cognition_v2 import sandbox  # noqa: E402

MARKER = ".cognition-v2-eval"
TOOLSETS = ["file", "terminal", "cognition_wm"]
ARMS = {
    "off": {"plugin": False, "evidence_loop": False, "learning": False},  # the existing cognition
    "on": {"plugin": True, "evidence_loop": True, "learning": False},  # phase 1
    "v2": {"plugin": True, "evidence_loop": True, "learning": True},  # phase 1 + learning and calibration
}
METRICS = ("wall_s", "api_calls", "tool_calls", "blocked_tool_calls", "prompt_tokens", "completion_tokens",
           "snapshot_chars", "lessons_chars", "evidence_rounds", "evidence_unsupported")
_LESSONS = re.compile(r'<lessons unchanged rev="[^"]+"/>|<lessons rev="[^"]+">[\s\S]*?</lessons>')


def _borrows_logins(home: Path) -> bool:
    """Whether the home would borrow the Codex CLI / Claude Code login (``auth.adopt_external_logins``,
    on by default): their refresh tokens rotate, so a borrowed login can log those programs out."""
    from hermes_cli.config import read_user_config_raw

    path = home / "config.yaml"
    auth = (read_user_config_raw(path) if path.exists() else {}).get("auth")
    return not isinstance(auth, dict) or auth.get("adopt_external_logins", True) is not False


def init_home(home: Path) -> None:
    from hermes_cli.config import atomic_config_write, read_user_config_raw

    home.mkdir(parents=True, exist_ok=True)
    (home / MARKER).write_text("cognition v2 eval home — safe to delete\n", encoding="utf-8")
    path = home / "config.yaml"
    data = read_user_config_raw(path) if path.exists() else {}
    data.setdefault("auth", {})["adopt_external_logins"] = False  # only the login made in this home
    atomic_config_write(path, data)
    print(f"eval home ready: {home}\n"
          f"log in there yourself: HERMES_HOME={home} hermes auth add openai-codex --type oauth\n"
          "(answer N if it offers to import the Codex CLI credentials)")


def _check_home(home: Path) -> None:
    everyday = (Path.home() / ".hermes").resolve()
    resolved = home.resolve()
    if resolved == everyday or everyday in resolved.parents:
        sys.exit(f"refusing to run in the everyday home {everyday}; use a dedicated eval home")
    if not (resolved / MARKER).exists():
        sys.exit(f"{home} is not an eval home; create one with --init-home")
    if _borrows_logins(resolved):
        sys.exit(f"{home} could borrow another program's login; set auth.adopt_external_logins: false "
                 "(--init-home does)")


def _set_arm(home: Path, arm: dict) -> None:
    """Point the eval home's config at one arm through the canonical config writer."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw

    path = home / "config.yaml"
    data = read_user_config_raw(path) if path.exists() else {}
    plugins = data.setdefault("plugins", {})
    enabled = [p for p in plugins.get("enabled") or [] if p != "cognition_wm"]
    plugins["enabled"] = enabled + (["cognition_wm"] if arm["plugin"] else [])
    settings = plugins.setdefault("entries", {}).setdefault("cognition_wm", {}).setdefault("settings", {})
    settings["learning"] = settings["calibration"] = arm["learning"]
    cognition = data.setdefault("cognition", {})
    cognition["enabled"] = True
    cognition.setdefault("evidence_loop", {})["enabled"] = arm["evidence_loop"]
    atomic_config_write(path, data)


def _turn_rows(messages: list) -> tuple[dict, list]:
    start = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    return messages[start], messages[start:]


def _snapshot_chars(user_row: dict) -> int:
    content, api = user_row.get("content"), user_row.get("api_content")
    return max(0, len(api) - len(content) - 2) if isinstance(api, str) and isinstance(content, str) else 0


def _write_files(work: Path, files: dict | None) -> None:
    for name, text in (files or {}).items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(text, encoding="utf-8")


def _lessons_chars(user_row: dict) -> int:
    api = user_row.get("api_content")
    return sum(len(found) for found in _LESSONS.findall(api)) if isinstance(api, str) else 0


def _ran(rows: list) -> list[str]:
    """The shell commands of the turn that ran (a blocked call did not)."""
    results = {m.get("tool_call_id"): str(m.get("content")) for m in rows if m.get("role") == "tool"}
    commands = []
    for message in rows:
        for call in (message.get("tool_calls") or []) if message.get("role") == "assistant" else []:
            function = call.get("function") or {}
            if function.get("name") != "terminal":
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if "BLOCKED" not in results.get(call.get("id"), ""):
                commands.append(str(args.get("command") or ""))
    return commands


def _file_text(work: Path, name: str) -> str:
    path = work / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


# check kind -> whether it holds, given (its value, the check, the turn's context)
_CHECKS = {
    "ran": lambda value, _check, ctx: any(value in command for command in ctx["ran"]),
    "not_ran": lambda value, _check, ctx: not any(value in command for command in ctx["ran"]),
    "answer_has": lambda value, _check, ctx: value in ctx["answer"],
    "answer_lacks": lambda value, _check, ctx: value not in ctx["answer"],
    "file_has": lambda value, check, ctx: value in _file_text(ctx["work"], check["file"]),
    "file_lacks": lambda value, check, ctx: value not in _file_text(ctx["work"], check["file"]),
    "file_exists": lambda value, check, ctx: (ctx["work"] / check["file"]).exists() == bool(value),
}


def _checks(case: dict, session: int, turn: int, ctx: dict) -> list[dict]:
    """The case's machine-checkable expectations for this turn, each with whether it held."""
    found = []
    for check in case.get("expect") or []:
        if (check.get("session", 0), check.get("turn", 0)) != (session, turn):
            continue
        kind = next(k for k in _CHECKS if k in check)
        found.append({"check": f"{kind}: {check[kind]}" + (f" ({check['file']})" if "file" in check else ""),
                      "ok": bool(_CHECKS[kind](check[kind], check, ctx))})
    return found


def _sessions(case: dict) -> list[dict]:
    """A case's conversations; a phase-1 case (``turns``) is one conversation in the case directory."""
    if case.get("sessions"):
        return case["sessions"]
    return [{"turns": case["turns"], "files_before_turn": case.get("files_before_turn"),
             "compact_before_turn": case.get("compact_before_turn")}]


def _learning_store():
    from plugins.cognition_wm.experience_store import ExperienceStore

    return ExperienceStore.open()


def _start_clean() -> None:
    """Nothing learned or recorded by an earlier case run carries into this one."""
    store = _learning_store()
    store.forget_experiences()
    store.forget_outcomes()


def _seed_outcomes(case: dict, home: Path) -> None:
    """Earlier results a case starts from (a track record), for the eval model in the case directory."""
    if not case.get("seed_outcomes"):
        return
    from hermes_cli import __version__
    from hermes_cli.config import read_user_config_raw
    from plugins.cognition_wm import runtime

    model = (read_user_config_raw(home / "config.yaml").get("model") or {})
    store, now = _learning_store(), time.time()
    for spec in case["seed_outcomes"]:
        for n, result in enumerate(spec["results"]):
            store.record_outcome(evidence_key=f"seed:{case['id']}:{spec['claim_class']}:{n}", session_id="seed",
                                 turn_id=f"seed-{n}", task_id=None, project_key=runtime.project_key(),
                                 task_kind=spec["task_kind"], claim_class=spec["claim_class"], actor_kind="self",
                                 actor_id=str(model.get("default") or ""), provider=str(model.get("provider") or ""),
                                 hermes_version=__version__, result=result, at=now - 3600 + n)


def _case_summary(case: dict, arm_name: str, repeat: int) -> dict:
    store, lessons, claims = _learning_store(), {}, {}
    for lesson in store.experiences():
        lessons[lesson.status] = lessons.get(lesson.status, 0) + 1
    for outcome in store.outcomes():
        results = claims.setdefault(outcome.actor_kind, {})
        results[outcome.result] = results.get(outcome.result, 0) + 1
    return {"kind": "case", "case": case["id"], "set": case.get("set", "dev"), "arm": arm_name, "repeat": repeat,
            "lessons": lessons, "outcomes": claims}


def _run_case(case: dict, arm_name: str, repeat: int, work_root: Path, home: Path) -> Iterable[dict]:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    root = work_root / f"{case['id']}-{arm_name}-{repeat}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    _write_files(root, case.get("files"))
    os.chdir(root)
    _start_clean()
    _seed_outcomes(case, home)
    db = SessionDB()
    stamp = int(time.time() * 1000)
    for s_index, session in enumerate(_sessions(case)):
        work = root / session["project"] if session.get("project") else root
        work.mkdir(parents=True, exist_ok=True)
        _write_files(work, session.get("files"))
        os.chdir(work)
        session_id = f"cv2-{case['id']}-{arm_name}-{repeat}-{s_index}-{stamp}"
        for index, message in enumerate(session["turns"]):
            _write_files(work, (session.get("files_before_turn") or {}).get(str(index)))  # the world changes
            if session.get("compact_before_turn") == index:
                _compress(AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                                  enabled_toolsets=TOOLSETS), db, session_id)
            agent = AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                            enabled_toolsets=TOOLSETS)
            started = time.monotonic()
            result = agent.run_conversation(message, conversation_history=db.get_messages_as_conversation(session_id))
            wall = time.monotonic() - started
            user_row, rows = _turn_rows(result.get("messages") or [])
            tools = [m for m in rows if m.get("role") == "tool"]
            calls = [c.get("function") or {} for m in rows if m.get("role") == "assistant"
                     for c in m.get("tool_calls") or []]
            meta = getattr(agent, "_current_turn_cognition_metadata", None) or {}
            evidence = (meta.get("cognition_trace") or {}).get("evidence") or {}
            answer = str(result.get("final_response") or "")
            yield {
                "kind": "turn", "case": case["id"], "set": case.get("set", "dev"), "arm": arm_name, "repeat": repeat,
                "session": s_index, "turn": index, "model": agent.model, "provider": agent.provider,
                "wall_s": round(wall, 3), "api_calls": result.get("api_calls"),
                "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
                "estimated_cost_usd": result.get("estimated_cost_usd"),
                "tool_calls": len(tools), "blocked_tool_calls": sum("BLOCKED" in str(m.get("content")) for m in tools),
                "snapshot_chars": _snapshot_chars(user_row), "lessons_chars": _lessons_chars(user_row),
                "evidence_rounds": evidence.get("rounds_used", 0),
                "evidence_unsupported": len(evidence.get("unsupported") or []),
                "evidence_downgraded": evidence.get("downgraded", False),
                "answer_checked": bool(meta.get("verification_applied")),
                "track_record_band": meta.get("track_record_band"),
                "checks": _checks(case, s_index, index, {"ran": _ran(rows), "answer": answer, "work": work}),
                "final_response": answer[:800],
                # every call the model made, for review: a confined run should show nothing outside the case
                "calls": [f"{c.get('name')} {str(c.get('arguments') or '')[:300]}" for c in calls],
            }
    yield _case_summary(case, arm_name, repeat)


def _compress(agent: Any, db: Any, session_id: str) -> None:
    from tui_gateway.session_compression import _compress_session_history

    _compress_session_history({"agent": agent, "history": db.get_messages_as_conversation(session_id),
                               "history_lock": threading.Lock(), "history_version": 1, "session_key": session_id}, "")


def run(home: Path, cases_path: Path, out: Path, *, arms: list[str], repeats: int, case_set: str) -> None:
    _check_home(home)
    os.environ["HERMES_HOME"] = str(home.resolve())
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.plugins import discover_plugins

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [c for c in cases if case_set == "all" or c.get("set", "dev") == case_set]
    original = read_user_config_raw(home / "config.yaml") if (home / "config.yaml").exists() else None
    try:
        with out.open("a", encoding="utf-8") as sink:
            for arm_name in arms:
                _set_arm(home, ARMS[arm_name])
                discover_plugins(force=True)
                for repeat in range(repeats):
                    for case in cases:
                        for row in _run_case(case, arm_name, repeat, home / "eval-work", home):
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            if row["kind"] == "turn":
                                print(f"{row['arm']} r{repeat} {row['case']} s{row['session']} t{row['turn']}: "
                                      f"{row['wall_s']}s calls={row['api_calls']} blocked={row['blocked_tool_calls']} "
                                      f"checks={sum(c['ok'] for c in row['checks'])}/{len(row['checks'])}")
    finally:
        if original is not None:
            from hermes_cli.config import atomic_config_write

            atomic_config_write(home / "config.yaml", original)


def summarize(path: Path) -> dict:
    """Per arm and metric: median and range over the turn rows, the share of checks that held, and
    how often the answer check ran (no thresholds are applied here)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    turns = [row for row in rows if row.get("kind", "turn") == "turn"]
    by_arm: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in turns:
        for metric in METRICS:
            if isinstance(row.get(metric), (int, float)):
                by_arm[row["arm"]][metric].append(row[metric])
    summary = {arm: {m: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
                     for m, v in metrics.items()} for arm, metrics in by_arm.items()}
    for arm in summary:
        mine = [row for row in turns if row["arm"] == arm]
        checks = [c["ok"] for row in mine for c in row.get("checks") or []]
        summary[arm]["checks"] = {"held": sum(checks), "total": len(checks)}
        summary[arm]["answer_checked"] = {"turns": sum(bool(row.get("answer_checked")) for row in mine),
                                          "of": len(mine)}
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init-home", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arms", default="off,on,v2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--set", dest="case_set", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args(argv)
    if args.init_home:
        init_home(args.init_home)
    elif args.summarize:
        print(json.dumps(summarize(args.summarize), ensure_ascii=False, indent=2))
    elif args.home and args.cases and args.out:
        if os.environ.get(sandbox.MARKER_ENV) != "1":
            _check_home(args.home)
            if not sandbox.available():
                sys.exit("the real-model evaluation runs only confined (macOS sandbox-exec); none is available here")
            options = ["--arms", args.arms, "--repeats", str(args.repeats), "--set", args.case_set]
            sys.exit(sandbox.rerun_confined(eval_home=args.home, cases=args.cases, out=args.out, options=options))
        run(args.home, args.cases, args.out, arms=args.arms.split(","), repeats=args.repeats, case_set=args.case_set)
    else:
        parser.error("use --init-home, --summarize, or --home/--cases/--out")


if __name__ == "__main__":
    main()
