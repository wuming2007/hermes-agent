"""Confinement for the real-model evaluation (macOS Seatbelt, ``sandbox-exec``).

A real model gets terminal and file tools here, and a separate HERMES_HOME and work directory are
not a sandbox: it could still ``cd ~`` and change real files, start a real coding agent with the
user's login, or talk to launchd. So the run is re-executed under a Seatbelt profile that:

- denies reading and writing the real home and the data volume, except the eval home (config,
  login, work directories), this checkout (the code and its venv) and the Python install;
- gives the run its own HOME inside the eval home, and a PATH whose ``claude``/``codex``/...
  are stand-ins that only log the call;
- denies running the real coding agents, ``launchctl``, ``osascript``, ``open``, ``crontab``,
  ``sudo``, the keychain CLI, ``hermes`` and the usual network clients.

What stays open: network (the model's own API) and reading the eval home, including its login.
The report lists every command a case ran, so a run that reached for either shows up there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

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


def profile(*, eval_home: Path, repo: Path, home: Path, python_root: Path, denied: Iterable[Path],
            protected: Iterable[Path] = PROTECTED, readable: Iterable[Path] = (), writable: Iterable[Path] = ()) -> str:
    """The Seatbelt profile (later rules win)."""
    rules = ["(version 1)", "(allow default)"]
    rules += [f'(deny file-read* file-write* (subpath "{_sb(area)}"))' for area in (home, *protected)]
    rules.append(f'(allow file-read-metadata (subpath "{_sb(home)}"))')  # stat for path resolution, no contents
    rules += [f'(allow file-read* (subpath "{_sb(path)}"))' for path in (repo, python_root)]
    rules += [f'(allow file-read* (literal "{_sb(path)}"))' for path in readable]
    rules += [f'(allow file-read* file-write* (literal "{_sb(path)}"))' for path in writable]
    rules.append(f'(allow file-read* file-write* (subpath "{_sb(eval_home)}"))')
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


def stand_ins(eval_home: Path) -> Path:
    """A bin directory whose coding-agent commands only log their call to ``calls.log``."""
    bin_dir = eval_home / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in AGENT_CLIS:
        script = bin_dir / name
        script.write_text(_STAND_IN.format(log=bin_dir / "calls.log"), encoding="utf-8")
        script.chmod(0o755)
    return bin_dir


def rerun_confined(*, eval_home: Path, cases: Path, out: Path, options: list[str]) -> int:
    """Run this evaluation again under the profile (paths made absolute, since the confined run
    works in the eval home); returns its exit status."""
    eval_home, cases, out = eval_home.resolve(), cases.resolve(), out.resolve()
    repo = Path(__file__).resolve().parents[2]
    real_home = Path.home().resolve()
    python_root = Path(sys.executable).resolve().parents[1]
    bin_dir = stand_ins(eval_home)
    (eval_home / "home").mkdir(exist_ok=True)
    denied = _installed((*AGENT_CLIS, *DENIED_TOOLS), os.environ.get("PATH", "") + os.pathsep + "/usr/bin:/bin:/usr/sbin")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    text = profile(eval_home=eval_home, repo=repo, home=real_home, python_root=python_root, denied=denied,
                   readable=(cases,), writable=(out,))
    profile_path = eval_home / "sandbox.sb"
    profile_path.write_text(text, encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "HOME": str(eval_home / "home"),
        "HERMES_HOME": str(eval_home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": os.environ.get("TERM", "dumb"),
        MARKER_ENV: "1",
    }
    command = [str(SANDBOX_EXEC), "-f", str(profile_path), sys.executable, str(Path(__file__).with_name("runner.py")),
               "--home", str(eval_home), "--cases", str(cases), "--out", str(out), *options]
    return subprocess.run(command, env=env, cwd=str(eval_home), check=False).returncode


def command_log(eval_home: Path) -> list[str]:
    """The coding-agent commands the run tried, as the stand-ins logged them."""
    log = eval_home / "fake-bin" / "calls.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []
