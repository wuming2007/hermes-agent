"""Cognition v2 phase 1 evaluation: the same cases, OFF against ON, with the model of an eval home.

OFF is today's behavior (cognition routing on, no working memory, no evidence loop); ON enables
the cognition_wm plugin and the evidence loop. Each case is a short multi-turn conversation run
as a fresh agent per turn (like ``hermes chat -r``) in its own work directory; each turn writes
one JSONL row: latency, API calls, tokens, tool calls and how many were blocked, the snapshot
size, the evidence-loop outcome and the final answer.

Runs only in a dedicated eval home made with ``--init-home`` (never the everyday ``~/.hermes``).
The toolsets are file, terminal and task_state: no messaging, web, cron or subagents, so a case
cannot reach anyone. The user logs in to the eval home themselves; nothing copies credentials,
and the eval home never borrows the Codex CLI or Claude Code login (``auth.adopt_external_logins``
is turned off by ``--init-home`` and checked before every run).

    python evals/cognition_v2/runner.py --init-home ~/hermes-eval-home
    HERMES_HOME=~/hermes-eval-home hermes auth add openai-codex --type oauth   # the user, once
    python evals/cognition_v2/runner.py --home ~/hermes-eval-home \\
        --cases evals/cognition_v2/cases.example.jsonl --out /tmp/cv2.jsonl --set dev --repeats 3
    python evals/cognition_v2/runner.py --summarize /tmp/cv2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MARKER = ".cognition-v2-eval"
TOOLSETS = ["file", "terminal", "cognition_wm"]
ARMS = {
    "off": {"plugin": False, "evidence_loop": False},
    "on": {"plugin": True, "evidence_loop": True},
}
METRICS = ("wall_s", "api_calls", "tool_calls", "blocked_tool_calls", "prompt_tokens", "completion_tokens",
           "snapshot_chars", "evidence_rounds", "evidence_unsupported")


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


def _run_case(case: dict, arm_name: str, repeat: int, work_root: Path) -> Iterable[dict]:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    work = work_root / f"{case['id']}-{arm_name}-{repeat}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    _write_files(work, case.get("files"))
    os.chdir(work)
    db = SessionDB()
    session_id = f"cv2-{case['id']}-{arm_name}-{repeat}-{int(time.time())}"
    for index, message in enumerate(case["turns"]):
        _write_files(work, (case.get("files_before_turn") or {}).get(str(index)))  # the world changes between turns
        if case.get("compact_before_turn") == index:
            _compress(AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                              enabled_toolsets=TOOLSETS), db, session_id)
        agent = AIAgent(session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
                        enabled_toolsets=TOOLSETS)
        started = time.monotonic()
        result = agent.run_conversation(message, conversation_history=db.get_messages_as_conversation(session_id))
        wall = time.monotonic() - started
        user_row, rows = _turn_rows(result.get("messages") or [])
        tools = [m for m in rows if m.get("role") == "tool"]
        evidence = ((getattr(agent, "_current_turn_cognition_metadata", None) or {}).get("cognition_trace") or {}
                    ).get("evidence") or {}
        yield {
            "case": case["id"], "set": case.get("set", "dev"), "arm": arm_name, "repeat": repeat, "turn": index,
            "model": agent.model, "provider": agent.provider, "wall_s": round(wall, 3),
            "api_calls": result.get("api_calls"), "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "tool_calls": len(tools), "blocked_tool_calls": sum("BLOCKED" in str(m.get("content")) for m in tools),
            "snapshot_chars": _snapshot_chars(user_row),
            "evidence_rounds": evidence.get("rounds_used", 0),
            "evidence_unsupported": len(evidence.get("unsupported") or []),
            "evidence_downgraded": evidence.get("downgraded", False),
            "final_response": str(result.get("final_response") or "")[:800],
        }


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
                        for row in _run_case(case, arm_name, repeat, home / "eval-work"):
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            print(f"{row['arm']} r{repeat} {row['case']} t{row['turn']}: {row['wall_s']}s "
                                  f"calls={row['api_calls']} blocked={row['blocked_tool_calls']}")
    finally:
        if original is not None:
            from hermes_cli.config import atomic_config_write

            atomic_config_write(home / "config.yaml", original)


def summarize(path: Path) -> dict:
    """Per arm and metric: median and range over all rows (no thresholds are applied here)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_arm: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            if isinstance(row.get(metric), (int, float)):
                by_arm[row["arm"]][metric].append(row[metric])
    return {arm: {m: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
                  for m, v in metrics.items()} for arm, metrics in by_arm.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init-home", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arms", default="off,on")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--set", dest="case_set", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args(argv)
    if args.init_home:
        init_home(args.init_home)
    elif args.summarize:
        print(json.dumps(summarize(args.summarize), ensure_ascii=False, indent=2))
    elif args.home and args.cases and args.out:
        run(args.home, args.cases, args.out, arms=args.arms.split(","), repeats=args.repeats, case_set=args.case_set)
    else:
        parser.error("use --init-home, --summarize, or --home/--cases/--out")


if __name__ == "__main__":
    main()
