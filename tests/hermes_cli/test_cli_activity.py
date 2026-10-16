"""CLI activity publisher + reader (``hermes_cli.cli_activity``)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import cli_activity
from hermes_cli.cli_activity import CliActivityPublisher, read_activities


def _record(pub: CliActivityPublisher) -> dict:
    return json.loads(pub.path.read_text(encoding="utf-8"))


def _state(pub: CliActivityPublisher) -> tuple[str, str | None]:
    act = _record(pub)["activity"]
    return act["state"], act.get("waiting_for")


def test_turn_lifecycle_derives_the_pet_priority_and_bumps_seq_only_on_change(tmp_path: Path) -> None:
    pub = CliActivityPublisher(tmp_path, profile="default")
    run_id = pub.turn_started("sess-1")
    assert _state(pub) == ("thinking", None)
    seq = _record(pub)["seq"]
    pub.model_call_started()  # already thinking: no rewrite
    assert _record(pub)["seq"] == seq

    pub.tool_started("terminal")
    pub.tool_started("read_file")  # parallel tools
    assert _state(pub) == ("working", None)
    assert _record(pub)["activity"]["tool"] == "read_file"
    with pub.waiting("approval"):
        assert _state(pub) == ("waiting", "approval")  # waiting outranks a running tool
    assert _state(pub) == ("working", None)
    pub.tool_finished()
    assert _state(pub)[0] == "working"  # one tool still running
    pub.tool_finished()
    assert _state(pub)[0] == "thinking"
    pub.answer_text()
    assert _state(pub)[0] == "talking"
    pub.segment_boundary()
    assert _state(pub)[0] == "thinking"

    pub.turn_finished({"completed": True}, session_id="sess-2")
    rec = _record(pub)
    assert rec["activity"]["state"] == "completed" and rec["activity"]["run_id"] == run_id
    assert rec["activity"]["session_id"] == "sess-2"  # compression rotation followed
    end_seq = rec["seq"]
    pub.turn_finished({"failed": True, "error": "late"})  # only the first end counts
    pub.turn_cancelled("late")
    assert _record(pub)["seq"] == end_seq and _state(pub)[0] == "completed"
    assert stat.S_IMODE(pub.path.stat().st_mode) == 0o600


@pytest.mark.parametrize(("result", "state"), [
    ({"completed": True}, "completed"),
    ({"interrupted": True, "completed": False}, "cancelled"),
    ({"failed": True, "completed": False, "error": "boom"}, "failed"),
    ({"partial": True, "completed": False}, "failed"),
    (None, "failed"),
])
def test_turn_result_flags_map_to_outcomes(tmp_path: Path, result, state) -> None:
    pub = CliActivityPublisher(tmp_path)
    pub.turn_started("s")
    pub.turn_finished(result)
    assert _state(pub)[0] == state


def test_record_carries_identity_but_no_content(tmp_path: Path) -> None:
    pub = CliActivityPublisher(tmp_path, profile="work")
    pub.turn_started("sess")
    pub.tool_started("terminal")
    rec = _record(pub)
    assert rec["profile"] == "work" and rec["pid"] == os.getpid() and rec["surface"] == "cli"
    assert set(rec["activity"]) == {"run_id", "session_id", "state", "tool", "waiting_for", "error",
                                    "started_at", "updated_at", "ended_at"}


def test_close_mid_turn_is_cancelled_and_a_finished_outcome_survives_exit(tmp_path: Path) -> None:
    busy = CliActivityPublisher(tmp_path)
    busy.turn_started("s")
    busy.close()
    assert _state(busy)[0] == "cancelled"
    done = CliActivityPublisher(tmp_path)
    done.turn_started("s")
    done.turn_finished({"completed": True})
    done.close()  # e.g. `hermes chat -q` exits right after its only turn
    assert _state(done)[0] == "completed"
    never = CliActivityPublisher(tmp_path)
    never._write()
    never.close()
    assert not never.path.exists()


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write(home: Path, name: str, **over) -> Path:
    now = time.time()
    record = {"schema": 1, "publisher_id": name, "surface": "cli", "profile": "default", "pid": os.getpid(),
              "process_start_time": None, "seq": 3, "updated_at": now,
              "activity": {"run_id": f"run-{name}", "session_id": "s", "state": "thinking", "tool": None,
                           "waiting_for": None, "error": None, "started_at": now, "updated_at": now,
                           "ended_at": None}}
    act = over.pop("activity", {})
    record.update(over)
    record["activity"].update(act)
    path = cli_activity.activity_dir(home) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_reader_reports_a_dead_process_mid_turn_as_lost_not_idle(tmp_path: Path) -> None:
    _write(tmp_path, "alive")
    _write(tmp_path, "dead", pid=_dead_pid(), activity={"state": "working", "tool": "terminal"})
    acts = {a["publisher_id"]: a for a in read_activities(tmp_path)}
    assert acts["alive"]["state"] == "thinking" and acts["alive"]["source_alive"] is True
    assert acts["dead"]["state"] == "lost" and acts["dead"]["source_alive"] is False
    assert acts["dead"]["tool"] is None


def test_reader_retention_and_garbage(tmp_path: Path) -> None:
    old = time.time() - 3 * cli_activity.LOST_RETENTION_SECONDS
    _write(tmp_path, "done-recent", activity={"state": "completed"})
    _write(tmp_path, "done-old", activity={"state": "completed", "updated_at": time.time() - 3600})
    _write(tmp_path, "lost-old", pid=_dead_pid(), activity={"updated_at": old})
    _write(tmp_path, "future-schema", schema=99)
    (cli_activity.activity_dir(tmp_path) / "torn.json").write_text("{not json", encoding="utf-8")
    assert [a["publisher_id"] for a in read_activities(tmp_path)] == ["done-recent"]


def test_new_publisher_sweeps_long_dead_records_only(tmp_path: Path) -> None:
    stale = _write(tmp_path, "stale", pid=_dead_pid(), updated_at=time.time() - 2 * cli_activity.SWEEP_AFTER_SECONDS)
    fresh_dead = _write(tmp_path, "fresh-dead", pid=_dead_pid())
    live = _write(tmp_path, "live", updated_at=time.time() - 2 * cli_activity.SWEEP_AFTER_SECONDS)
    CliActivityPublisher(tmp_path).turn_started("s")
    assert not stale.exists() and fresh_dead.exists() and live.exists()


def test_feed_can_be_disabled(monkeypatch) -> None:
    class Owner:
        pass
    monkeypatch.setenv("HERMES_CLI_ACTIVITY_FEED", "0")
    owner = Owner()
    assert cli_activity.publisher_for(owner) is None
    cli_activity.notify(owner, "tool_started", "x")  # no publisher: silently nothing
    with cli_activity.waiting(owner, "approval"):
        pass
