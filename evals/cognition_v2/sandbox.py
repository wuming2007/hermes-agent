"""Confinement for the real-model evaluation (macOS Seatbelt, ``sandbox-exec``).

A real model gets terminal and file tools here, and a separate HERMES_HOME and work directory are
not a sandbox: it could still ``cd ~`` and change real files, start a real coding agent with the
user's login, or talk to launchd, and it could read the cases with their checks, the results so
far, or what another case run left behind. So every case run is its own process, run under a
Seatbelt profile that:

- denies reading and writing the real home and the data volume, except this case run's own home
  (a fresh directory under the eval home, removed afterwards), this checkout (the code and its
  venv) and the Python install; the cases file, the results file and the other case files next to
  the cases are not readable even inside the checkout;
- gives the run its own HOME inside its home, and a PATH whose ``claude``/``codex``/... are
  stand-ins that only log the call;
- denies running the real coding agents, ``launchctl``, ``osascript``, ``open``, ``crontab``,
  ``sudo``, the keychain CLI, ``hermes`` and the usual network clients.

The case (with its checks) reaches the run on stdin from the trusted runner outside the profile,
and the rows come back on stdout, each after a one-time key the run was given, so nothing inside
can add a row of its own. What stays open: network (the model's own API) and reading the run's own
home, including a copy of the eval home's login, which the runner copies back when it changes.
The report lists every command a case ran.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

MARKER_ENV = "COGNITION_V2_EVAL_SANDBOXED"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
AGENT_CLIS = ("claude", "codex", "gemini", "opencode", "aider", "cursor-agent")
DENIED_TOOLS = ("launchctl", "osascript", "open", "crontab", "sudo", "security", "hermes", "curl", "wget", "nc",
                "ssh", "scp", "sftp", "rsync", "telnet", "ftp")
PROTECTED = (Path("/Volumes/Data"),)
_STAND_IN = """#!/bin/sh
echo "$(basename "$0") $*" >> "{log}"
echo "[cognition v2 eval] stand-in for $(basename "$0"): not run"
"""


def available() -> bool:
    return sys.platform == "darwin" and SANDBOX_EXEC.exists()


def _sb(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _sb_regex(path: Path) -> str:
    return "".join("\\" + c if c in ".^$*+?()[]{}|\\" else c for c in str(path)).replace('"', '\\"')


def profile(*, run_home: Path, repo: Path, home: Path, python_root: Path, denied: Iterable[Path],
            protected: Iterable[Path] = PROTECTED, walled: Iterable[Path] = (), unreadable: Iterable[Path] = ()) -> str:
    """The Seatbelt profile (later rules win). ``walled``: areas around the run's home (the eval
    home) that it may pass through but neither list nor read. ``unreadable``: files the run never
    reads, and every other ``.jsonl`` file next to them (the other case sets)."""
    walled = tuple(walled)
    rules = ["(version 1)", "(allow default)"]
    rules += [f'(deny file-read* file-write* (subpath "{_sb(area)}"))' for area in (home, *protected, *walled)]
    # stat for path resolution, no listing and no contents
    rules += [f'(allow file-read-metadata (subpath "{_sb(area)}"))' for area in (home, *walled)]
    rules += [f'(allow file-read* (subpath "{_sb(path)}"))' for path in (repo, python_root)]
    rules.append(f'(allow file-read* file-write* (subpath "{_sb(run_home)}"))')
    for path in unreadable:
        rules.append(f'(deny file-read* file-write* (literal "{_sb(path)}"))')
        rules.append(f'(deny file-read* file-write* (regex #"^{_sb_regex(path.parent)}/[^/]*\\.jsonl$"))')
    rules += [f'(deny process-exec (literal "{_sb(path)}"))' for path in denied]
    return "\n".join(rules) + "\n"


def _installed(names: Iterable[str], path_env: str) -> list[Path]:
    """Every copy of these commands on PATH, as found and as resolved (a symlink runs its target)."""
    found: list[Path] = []
    for directory in filter(None, path_env.split(os.pathsep)):
        for name in names:
            candidate = Path(directory) / name
            if candidate.exists():
                found += [candidate, candidate.resolve()]
    return list(dict.fromkeys(found))


def stand_ins(run_home: Path) -> Path:
    """A bin directory whose coding-agent commands only log their call to ``calls.log``."""
    bin_dir = run_home / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in AGENT_CLIS:
        script = bin_dir / name
        script.write_text(_STAND_IN.format(log=bin_dir / "calls.log"), encoding="utf-8")
        script.chmod(0o755)
    return bin_dir


def run_case(run_home: Path, case: dict, arm: str, repeat: int, *, eval_home: Path,
             unreadable: Iterable[Path] = ()) -> Iterator[dict]:
    """One case run in its own confined process, with ``run_home`` (inside ``eval_home``, the rest of
    which it cannot see) as its HERMES_HOME; yields its rows. The case goes in on stdin; a row counts
    only when it comes back after the one-time key."""
    run_home = run_home.resolve()
    repo = Path(__file__).resolve().parents[2]
    python_root = Path(sys.executable).resolve().parents[1]
    bin_dir = stand_ins(run_home)
    (run_home / "home").mkdir(exist_ok=True)
    denied = _installed((*AGENT_CLIS, *DENIED_TOOLS), os.environ.get("PATH", "") + os.pathsep + "/usr/bin:/bin:/usr/sbin")
    profile_path = run_home / "sandbox.sb"
    profile_path.write_text(profile(run_home=run_home, repo=repo, home=Path.home().resolve(), python_root=python_root,
                                    denied=denied, walled=(eval_home.resolve(),),
                                    unreadable=[Path(p).resolve() for p in unreadable]), encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "HOME": str(run_home / "home"),
        "HERMES_HOME": str(run_home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": os.environ.get("TERM", "dumb"),
        MARKER_ENV: "1",
    }
    key = secrets.token_hex(16)
    command = [str(SANDBOX_EXEC), "-f", str(profile_path), sys.executable, str(Path(__file__).with_name("runner.py")),
               "--run-case"]
    with subprocess.Popen(command, env=env, cwd=str(run_home), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8") as child:
        child.stdin.write(json.dumps({"key": key, "case": case, "arm": arm, "repeat": repeat}, ensure_ascii=False))
        child.stdin.close()
        ended = False
        for line in child.stdout:
            found, _, row = line.rstrip("\n").partition("\t")
            if found == key:
                row = json.loads(row)
                ended = ended or row.get("kind") in ("case", "error")
                yield row
            else:
                print(line, end="", file=sys.stderr)  # the run's own output, never a row
        status = child.wait()
    if not ended:
        yield {"kind": "error", "case": case["id"], "set": case.get("set", "dev"), "arm": arm, "repeat": repeat,
               "error": f"the case run ended without a summary (exit status {status})"}


def command_log(home: Path) -> list[str]:
    """The coding-agent commands the runs tried, as the stand-ins logged them."""
    log = home / "fake-bin" / "calls.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []
