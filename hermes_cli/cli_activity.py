"""Cross-process CLI turn activity: the CLI publishes, the gateway's ``/v1/activity`` reads.

One interactive or ``-q`` CLI process owns one JSON record under
``HERMES_HOME/runtime/activity/<publisher_id>.json`` (0600, atomic replace). It is rewritten
only on a real lifecycle transition the CLI itself observes (turn start, model call, tool
start/finish, first visible answer text, a blocking approval/clarify/secret prompt, turn end),
never per token, so a reader sees what the CLI is doing without parsing its terminal output.

Activity only: no message content, tool arguments or answers are written. Approvals and
answers stay with the CLI; a reader can show "waiting" but cannot respond.

Identity: ``publisher_id`` names the process (pid + random suffix); ``run_id`` names one turn;
``seq`` increases on every write, so a reader can drop a stale view. The record carries the
process ``pid`` + ``process_start_time``: a reader that finds the process gone while a turn is
still active reports ``lost`` instead of letting it pass for idle.

Disable with ``HERMES_CLI_ACTIVITY_FEED=0``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

SCHEMA = 1
ACTIVE_STATES = frozenset({"thinking", "working", "talking", "waiting"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
# A finished turn stays visible this long so a client that reconnects learns its outcome.
TERMINAL_RETENTION_SECONDS = 60.0
# A turn whose process died mid-turn is reported as ``lost`` for this long, then dropped.
LOST_RETENTION_SECONDS = 600.0
# Publisher files of dead processes older than this are swept by the next publisher.
SWEEP_AFTER_SECONDS = 3600.0
_ERROR_CHARS = 300


def activity_dir(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "runtime" / "activity"


def feed_enabled() -> bool:
    return os.environ.get("HERMES_CLI_ACTIVITY_FEED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "default"


class CliActivityPublisher:
    """Thread-safe writer for this process's activity record.

    Callbacks fire on the agent thread, the UI thread and atexit; every public method takes
    the lock, updates the in-memory record and, only when the derived state or detail changed,
    rewrites the file. Failures are logged at debug level and never reach the caller.
    """

    def __init__(self, hermes_home: str | Path, *, surface: str = "cli", profile: Optional[str] = None) -> None:
        from hermes_cli.active_sessions import _own_start_time

        self._dir = activity_dir(hermes_home)
        self.publisher_id = f"{surface}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._path = self._dir / f"{self.publisher_id}.json"
        self._lock = threading.RLock()
        self._seq = 0
        self._base = {
            "schema": SCHEMA, "publisher_id": self.publisher_id, "surface": surface,
            "profile": profile or _profile_name(), "pid": os.getpid(),
            "process_start_time": _own_start_time(),
        }
        self._turn: Optional[dict[str, Any]] = None
        # Signals the derived state is computed from (mirrors agent.pet.state priorities).
        self._tools = 0
        self._tool: Optional[str] = None
        self._waiting: list[str] = []
        self._talking = False
        self._closed = False
        self._swept = False

    @property
    def path(self) -> Path:
        return self._path

    # -- turn lifecycle ------------------------------------------------------------------
    def turn_started(self, session_id: Optional[str]) -> str:
        with self._lock:
            now = time.time()
            run_id = f"cli_{uuid.uuid4().hex}"
            self._tools = 0
            self._tool = None
            self._waiting = []
            self._talking = False
            self._turn = {"run_id": run_id, "session_id": str(session_id or ""), "state": "thinking",
                          "tool": None, "waiting_for": None, "error": None,
                          "started_at": now, "updated_at": now, "ended_at": None}
            self._sweep_dead_once()
            self._write()
            return run_id

    def turn_finished(self, result: Optional[dict[str, Any]], *, session_id: Optional[str] = None,
                      error: Optional[str] = None) -> None:
        """Map the CLI's own turn result flags; idempotent (only the first end counts)."""
        if result is None:
            outcome, message = "failed", error or "turn ended without a result"
        elif result.get("interrupted"):
            outcome, message = "cancelled", None
        elif result.get("failed") or result.get("partial") or not result.get("completed", True):
            outcome = "failed"
            message = str(result.get("error") or result.get("failure_reason") or "turn failed")
        else:
            outcome, message = "completed", None
        self._finish(outcome, message, session_id)

    def turn_cancelled(self, reason: str) -> None:
        self._finish("cancelled", reason, None)

    def _finish(self, outcome: str, message: Optional[str], session_id: Optional[str]) -> None:
        with self._lock:
            turn = self._turn
            if turn is None or turn["state"] in TERMINAL_STATES:
                return
            now = time.time()
            turn.update(state=outcome, tool=None, waiting_for=None, updated_at=now, ended_at=now,
                        error=(message or None) and str(message)[:_ERROR_CHARS])
            if session_id:
                turn["session_id"] = str(session_id)
            self._tools, self._waiting, self._talking = 0, [], False
            self._write()

    # -- in-turn signals ------------------------------------------------------------------
    def model_call_started(self) -> None:
        """A new model request (thinking spinner): a fresh answer segment starts later."""
        with self._lock:
            self._talking = False
            self._refresh()

    def tool_started(self, name: Optional[str]) -> None:
        with self._lock:
            self._tools += 1
            self._talking = False
            if name:
                self._tool = str(name)[:120]
            self._refresh()

    def tool_finished(self) -> None:
        with self._lock:
            self._tools = max(0, self._tools - 1)
            self._refresh()

    def answer_text(self) -> None:
        """Visible answer text reached the terminal (first chunk of a segment)."""
        with self._lock:
            if self._talking:
                return
            self._talking = True
            self._refresh()

    def segment_boundary(self) -> None:
        with self._lock:
            self._talking = False
            self._refresh()

    def waiting(self, kind: str) -> "_Waiting":
        """Context manager around a prompt that blocks the turn on the user."""
        return _Waiting(self, kind)

    def _wait_enter(self, kind: str) -> None:
        with self._lock:
            self._waiting.append(kind)
            self._refresh()

    def _wait_exit(self, kind: str) -> None:
        with self._lock:
            with_kind = [i for i, k in enumerate(self._waiting) if k == kind]
            if with_kind:
                self._waiting.pop(with_kind[-1])
            self._refresh()

    # -- shutdown -------------------------------------------------------------------------
    def close(self, reason: str = "cli_exit") -> None:
        """Process exit: an unfinished turn is cancelled (the CLI is gone); a finished turn's record
        stays so a reader still sees its outcome (readers drop it after the retention window and the
        next publisher sweeps it); a process that never ran a turn removes its record."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            turn = self._turn
            if turn is not None and turn["state"] in ACTIVE_STATES:
                now = time.time()
                turn.update(state="cancelled", tool=None, waiting_for=None, error=reason,
                            updated_at=now, ended_at=now)
                self._write(force=True)
                return
            if turn is not None:
                return
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("activity record removal failed", exc_info=True)

    # -- internals ------------------------------------------------------------------------
    def _derived(self) -> tuple[str, Optional[str]]:
        if self._waiting:
            return "waiting", self._waiting[-1]
        if self._tools > 0:
            return "working", None
        if self._talking:
            return "talking", None
        return "thinking", None

    def _refresh(self) -> None:
        turn = self._turn
        if turn is None or turn["state"] in TERMINAL_STATES:
            return
        state, waiting_for = self._derived()
        tool = self._tool if state == "working" else None
        if (state, waiting_for, tool) == (turn["state"], turn.get("waiting_for"), turn.get("tool")):
            return
        turn.update(state=state, waiting_for=waiting_for, tool=tool, updated_at=time.time())
        self._write()

    def _write(self, *, force: bool = False) -> None:
        if self._closed and not force:
            return
        self._seq += 1
        record = dict(self._base, seq=self._seq, updated_at=time.time(),
                      activity=dict(self._turn) if self._turn is not None else None)
        try:
            from utils import atomic_json_write
            self._dir.mkdir(parents=True, exist_ok=True)
            atomic_json_write(self._path, record, indent=None, mode=0o600)
        except Exception:
            logger.debug("activity record write failed", exc_info=True)

    def _sweep_dead_once(self) -> None:
        """Remove long-dead publishers' files (another CLI that crashed long ago)."""
        if self._swept:
            return
        self._swept = True
        cutoff = time.time() - SWEEP_AFTER_SECONDS
        for path in _record_paths(self._dir):
            if path == self._path:
                continue
            record = _load(path)
            if record is None:
                continue
            if _alive(record) is False and float(record.get("updated_at") or 0) < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass


class _Waiting:
    def __init__(self, publisher: CliActivityPublisher, kind: str) -> None:
        self._publisher, self._kind = publisher, kind

    def __enter__(self) -> None:
        try:
            self._publisher._wait_enter(self._kind)
        except Exception:
            logger.debug("activity wait enter failed", exc_info=True)

    def __exit__(self, *_exc: object) -> None:
        try:
            self._publisher._wait_exit(self._kind)
        except Exception:
            logger.debug("activity wait exit failed", exc_info=True)


# -- reader (gateway side) ------------------------------------------------------------------

def _record_paths(directory: Path) -> Iterable[Path]:
    try:
        return sorted(p for p in directory.iterdir() if p.suffix == ".json" and not p.name.startswith("."))
    except FileNotFoundError:
        return []
    except OSError:
        logger.debug("activity dir listing failed", exc_info=True)
        return []


def _load(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data


def _alive(record: dict[str, Any]) -> Optional[bool]:
    from hermes_cli.active_sessions import _pid_liveness
    return _pid_liveness(record.get("pid"), record.get("process_start_time"))


def read_activities(hermes_home: str | Path, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Flatten every publisher record into client-facing activities.

    A live process reports its record as written. A process that is gone while its turn was
    active reports ``lost`` (``source_alive: false``) — never idle. Finished turns are kept for
    ``TERMINAL_RETENTION_SECONDS`` and lost ones for ``LOST_RETENTION_SECONDS``.
    """
    now = time.time() if now is None else now
    out: list[dict[str, Any]] = []
    for path in _record_paths(activity_dir(hermes_home)):
        record = _load(path)
        if record is None:
            continue
        turn = record.get("activity")
        if not isinstance(turn, dict) or not turn.get("run_id"):
            continue
        state = str(turn.get("state") or "")
        alive = _alive(record)
        updated = float(turn.get("updated_at") or record.get("updated_at") or 0)
        if state in TERMINAL_STATES:
            if now - updated > TERMINAL_RETENTION_SECONDS:
                continue
        elif state in ACTIVE_STATES:
            if alive is False:
                if now - updated > LOST_RETENTION_SECONDS:
                    continue
                state = "lost"
        else:
            continue
        out.append({
            "id": str(turn["run_id"]),
            "run_id": str(turn["run_id"]),
            "publisher_id": str(record.get("publisher_id") or path.stem),
            "surface": str(record.get("surface") or "cli"),
            "profile": str(record.get("profile") or ""),
            "session_id": str(turn.get("session_id") or ""),
            "state": state,
            "tool": turn.get("tool") if state == "working" else None,
            "waiting_for": turn.get("waiting_for") if state == "waiting" else None,
            "error": turn.get("error"),
            "started_at": turn.get("started_at"),
            "updated_at": updated,
            "seq": int(record.get("seq") or 0),
            "source_alive": alive is not False,
        })
    out.sort(key=lambda a: (float(a.get("started_at") or 0), a["id"]))
    return out


# -- CLI glue -------------------------------------------------------------------------------

def publisher_for(owner: Any) -> Optional[CliActivityPublisher]:
    """This CLI's publisher, created on its first turn (main thread); None when disabled.
    Never raises: the activity feed must not change how a turn runs."""
    pub = getattr(owner, "_cli_activity_publisher", None)
    if pub is not None:
        return pub or None
    pub = False
    if feed_enabled():
        try:
            import atexit
            from hermes_constants import get_hermes_home
            pub = CliActivityPublisher(get_hermes_home())
            atexit.register(pub.close)
        except Exception:
            logger.debug("activity publisher unavailable", exc_info=True)
            pub = False
    try:
        owner._cli_activity_publisher = pub
    except Exception:
        return None
    return pub or None


def current(owner: Any) -> Optional[CliActivityPublisher]:
    """The publisher if one exists; never creates one (safe from any thread)."""
    pub = getattr(owner, "_cli_activity_publisher", None)
    return pub or None


def notify(owner: Any, method: str, *args: Any, **kwargs: Any) -> None:
    """Call a publisher method if this owner has one; never raises."""
    pub = current(owner)
    if pub is None:
        return
    try:
        getattr(pub, method)(*args, **kwargs)
    except Exception:
        logger.debug("activity %s failed", method, exc_info=True)


class _NoWait:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def waiting(owner: Any, kind: str) -> Any:
    pub = current(owner)
    return pub.waiting(kind) if pub is not None else _NoWait()
