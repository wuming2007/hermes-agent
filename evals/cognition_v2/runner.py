"""Cognition v2 evaluation: the same cases in three arms, with the model of an eval home.

``off`` is the existing cognition (routing on, no working memory, no evidence loop); ``on`` is
phase 1 (the cognition_wm plugin and the evidence loop); ``v2`` is phase 1 plus learning and
calibration. A case is one or more conversations (``sessions``, like the same user on another
day), each turn run as a fresh agent (like ``hermes chat -r``) in the case's own directories; each
turn writes one JSONL row (latency, API calls, tokens, tool calls and how many were blocked, the
working-memory and lessons characters added, the evidence-loop outcome, whether the answer check
ran, the case's checks for that turn, the final answer), and each case run ends with a summary row
(what was learned, how the recorded claims held up).

Every case run has a home of its own: a fresh directory under the eval home with a copy of its
config (set to the arm), login and persona, removed when the run ends, so it starts with nothing
learned and leaves nothing for the next run to find. The cases, with their checks, are read here,
by this trusted runner; a case run gets only its own case, and cannot read the cases file, the
results file or another run's home (``sandbox.py``).

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
import uuid
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

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
# What a case run's home copies from the eval home (the config is then set to the arm).
_CARRIED = ("config.yaml", "auth.json", ".env", "SOUL.md")


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
    """Point a home's config at one arm through the canonical config writer."""
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


def _runtime_kwargs() -> dict:
    """The provider, model, API mode and credentials this home's config selects, resolved the way
    ``hermes chat -q`` does; without it an agent falls back to an unresolved default."""
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import resolve_reasoning_config

    cfg = load_config()
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model = str(model_cfg.get("default") or model_cfg.get("model") or "")
    runtime = resolve_runtime_provider(requested=model_cfg.get("provider") or None, target_model=model or None)
    return {"api_key": runtime.get("api_key"), "base_url": runtime.get("base_url"), "provider": runtime.get("provider"),
            "requested_provider": runtime.get("requested_provider"), "api_mode": runtime.get("api_mode"),
            "model": model, "credential_pool": runtime.get("credential_pool"),
            "reasoning_config": resolve_reasoning_config(cfg, model)}


def _learning_store():
    from plugins.cognition_wm.experience_store import ExperienceStore

    return ExperienceStore.open()


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
    found = store.outcomes()
    for outcome in found:
        results = claims.setdefault(outcome.actor_kind, {})
        results[outcome.result] = results.get(outcome.result, 0) + 1
    return {"kind": "case", "case": case["id"], "set": case.get("set", "dev"), "arm": arm_name, "repeat": repeat,
            "lessons": lessons, "outcomes": claims,
            # each recorded claim, for review: who, what, how it held up (first reading), and why
            "claims": [f"{o.actor_kind}:{o.actor_id} {o.claim_class} {o.result}"
                       + (f" (was {o.prior_result})" if o.prior_result else "") + (f" [{o.reason}]" if o.reason else "")
                       for o in found]}


def _run_case(case: dict, arm_name: str, repeat: int, home: Path) -> Iterable[dict]:
    """One case run in ``home``, its own HERMES_HOME; its directories and session ids say nothing
    about the case or the arm."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    root = home / "work"
    root.mkdir(parents=True, exist_ok=True)
    _write_files(root, case.get("files"))
    os.chdir(root)
    _seed_outcomes(case, home)
    db = SessionDB()
    runtime = _runtime_kwargs()
    run_id = uuid.uuid4().hex[:12]
    for s_index, session in enumerate(_sessions(case)):
        work = root / session["project"] if session.get("project") else root
        work.mkdir(parents=True, exist_ok=True)
        _write_files(work, session.get("files"))
        os.chdir(work)
        session_id = f"cv2-{run_id}-{s_index}"
        for index, message in enumerate(session["turns"]):
            _write_files(work, (session.get("files_before_turn") or {}).get(str(index)))  # the world changes
            if session.get("compact_before_turn") == index:
                _compress(AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                                  enabled_toolsets=TOOLSETS, **runtime), db, session_id)
            agent = AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                            enabled_toolsets=TOOLSETS, **runtime)
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
    from tui_gateway.server import _compress_session_history  # the facade binds what the sibling late-imports

    _compress_session_history({"agent": agent, "history": db.get_messages_as_conversation(session_id),
                               "history_lock": threading.Lock(), "history_version": 1, "session_key": session_id}, "")


def _run_home(home: Path, arm: dict) -> Path:
    """A fresh home for one case run: the eval home's config (set to the arm), login and persona."""
    run_home = home / "runs" / uuid.uuid4().hex[:12]
    run_home.mkdir(parents=True)
    for name in _CARRIED:
        if (home / name).exists():
            shutil.copy2(home / name, run_home / name)
    (run_home / MARKER).write_text("cognition v2 case run — removed when the run ends\n", encoding="utf-8")
    _set_arm(run_home, arm)
    return run_home


def _close_run_home(run_home: Path, home: Path) -> None:
    """Keep what the eval home needs from a finished run — a refreshed login (its refresh token may
    have rotated) and the coding-agent calls the stand-ins logged — and remove the rest."""
    login = run_home / "auth.json"
    if login.exists() and (not (home / "auth.json").exists()
                           or login.read_bytes() != (home / "auth.json").read_bytes()):
        staged = home / "auth.json.cv2-tmp"
        shutil.copy2(login, staged)
        os.replace(staged, home / "auth.json")
    calls = run_home / "fake-bin" / "calls.log"
    if calls.exists():
        (home / "fake-bin").mkdir(exist_ok=True)
        with (home / "fake-bin" / "calls.log").open("a", encoding="utf-8") as log:
            log.write(calls.read_text(encoding="utf-8"))
    shutil.rmtree(run_home, ignore_errors=True)


def _run_here(run_home: Path, case: dict, arm_name: str, repeat: int) -> Iterable[dict]:
    """A case run in this process, with ``run_home`` as HERMES_HOME (the tests; a confined case run
    does the same in its own process)."""
    from hermes_cli.plugins import discover_plugins

    before_home, before_cwd = os.environ.get("HERMES_HOME"), os.getcwd()
    os.environ["HERMES_HOME"] = str(run_home)
    try:
        discover_plugins(force=True)
        yield from _run_case(case, arm_name, repeat, run_home)
    finally:
        os.chdir(before_cwd)
        if before_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = before_home


def _error_row(case: dict, arm_name: str, repeat: int, exc: Exception) -> dict:
    return {"kind": "error", "case": case["id"], "set": case.get("set", "dev"), "arm": arm_name, "repeat": repeat,
            "error": f"{type(exc).__name__}: {exc}"}


def run(home: Path, cases_path: Path, out: Path, *, arms: list[str], repeats: int, case_set: str,
        execute: Callable[..., Iterable[dict]] = _run_here) -> None:
    """Every case run in a home of its own, through ``execute(run_home, case, arm, repeat)``."""
    _check_home(home)
    home = home.resolve()
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [c for c in cases if case_set == "all" or c.get("set", "dev") == case_set]
    with out.open("a", encoding="utf-8") as sink:
        for arm_name in arms:
            for repeat in range(repeats):
                for case in cases:
                    run_home = _run_home(home, ARMS[arm_name])
                    try:
                        for row in execute(run_home, case, arm_name, repeat):
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            if row["kind"] == "turn":
                                print(f"{row['arm']} r{repeat} {row['case']} s{row['session']} t{row['turn']}: "
                                      f"{row['wall_s']}s calls={row['api_calls']} "
                                      f"blocked={row['blocked_tool_calls']} "
                                      f"checks={sum(c['ok'] for c in row['checks'])}/{len(row['checks'])}",
                                      flush=True)
                            elif row["kind"] == "error":
                                print(f"{arm_name} r{repeat} {case['id']}: ERROR {row['error']}", flush=True)
                    except Exception as exc:  # one broken case is recorded; hours of other cases are kept
                        error = _error_row(case, arm_name, repeat, exc)
                        sink.write(json.dumps(error, ensure_ascii=False) + "\n")
                        sink.flush()
                        print(f"{arm_name} r{repeat} {case['id']}: ERROR {error['error']}", flush=True)
                    finally:
                        _close_run_home(run_home, home)


def _serve_case() -> None:
    """Inside the confinement: the one case run the runner sent on stdin; each row goes to stdout
    after the one-time key it came with. The stream is taken before the case runs: compacting imports
    the TUI gateway, which points sys.stdout at stderr for its own protocol."""
    out = sys.stdout
    job = json.loads(sys.stdin.read())
    key, case, arm_name, repeat = job["key"], job["case"], job["arm"], job["repeat"]
    try:
        for row in _run_case(case, arm_name, repeat, Path(os.environ["HERMES_HOME"])):
            print(f"{key}\t{json.dumps(row, ensure_ascii=False)}", file=out, flush=True)
    except Exception as exc:
        print(f"{key}\t{json.dumps(_error_row(case, arm_name, repeat, exc), ensure_ascii=False)}", file=out, flush=True)


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
    for arm in {row["arm"] for row in rows}:
        summary.setdefault(arm, {})
        summary[arm]["errors"] = sum(row.get("kind") == "error" and row["arm"] == arm for row in rows)
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
    parser.add_argument("--run-case", action="store_true", help=argparse.SUPPRESS)  # the confined child only
    args = parser.parse_args(argv)
    if args.init_home:
        init_home(args.init_home)
    elif args.summarize:
        print(json.dumps(summarize(args.summarize), ensure_ascii=False, indent=2))
    elif args.run_case:
        if os.environ.get(sandbox.MARKER_ENV) != "1":
            sys.exit("--run-case runs only inside the confinement the runner sets up")
        _serve_case()
    elif args.home and args.cases and args.out:
        if not sandbox.available():
            sys.exit("the real-model evaluation runs only confined (macOS sandbox-exec); none is available here")
        run(args.home, args.cases, args.out, arms=args.arms.split(","), repeats=args.repeats, case_set=args.case_set,
            execute=partial(sandbox.run_case, eval_home=args.home, unreadable=(args.cases, args.out)))
    else:
        parser.error("use --init-home, --summarize, or --home/--cases/--out")


if __name__ == "__main__":
    main()
