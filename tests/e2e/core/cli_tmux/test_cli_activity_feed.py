"""CLI turn activity reaches an API client through the gateway's ``/v1/activity`` feed.

A real interactive ``hermes chat --cli`` runs in a private tmux server; a real ``gateway.run``
(api_server only) serves the SAME isolated HERMES_HOME. Both talk to one scripted fake
provider. The test only reads the gateway's HTTP surface (snapshot + SSE), exactly what a
desktop client sees: no CLI stdout parsing, no injected events.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Hang, Text, ToolCall, write_hermes_home

pytestmark = [
    pytest.mark.skipif(shutil.which("tmux") is None, reason="needs tmux"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process tree"),
]

REPO_ROOT = Path(__file__).resolve().parents[4]
ACTIVE = {"thinking", "working", "talking", "waiting"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Scripted:
    """Fake-provider responder keyed by a token in the turn's last user message, so several CLIs
    can run at once; each request for a token takes its next step, which may block on a gate."""

    def __init__(self) -> None:
        self._steps: dict[str, list[tuple[threading.Event | None, Any]]] = {}
        self._lock = threading.Lock()

    def push(self, token: str, response: Any, gate: threading.Event | None = None) -> None:
        with self._lock:
            self._steps.setdefault(token, []).append((gate, response))

    def __call__(self, record: dict) -> Any:
        users = [m for m in record["body"].get("messages", []) if m.get("role") == "user"]
        last = json.dumps(users[-1].get("content") if users else "", ensure_ascii=False)
        with self._lock:
            token = next((t for t in self._steps if t in last), None)
            steps = self._steps.get(token) or []
            gate, response = steps.pop(0) if steps else (None, Text("default reply"))
        if gate is not None:
            gate.wait(120)
        return response


class Harness:
    def __init__(self, root: Path, llm: FakeLLMServer) -> None:
        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.project = root / "project"
        self.project.mkdir(parents=True, exist_ok=True)
        write_hermes_home(
            self.hermes_home, llm.base_url,
            extra_config=(f"terminal:\n  cwd: {self.project}\n"
                          "display:\n  streaming: true\n"
                          "approvals:\n  mode: manual\n  timeout: 120\n"))
        self.key = secrets.token_hex(32)
        self.port = _free_port()
        with open(self.hermes_home / ".env", "a", encoding="utf-8") as fh:
            fh.write(f"API_SERVER_ENABLED=true\nAPI_SERVER_KEY={self.key}\n"
                     f"API_SERVER_HOST=127.0.0.1\nAPI_SERVER_PORT={self.port}\n")
        self.base = f"http://127.0.0.1:{self.port}"
        self.sock = f"hermes-act-{uuid.uuid4().hex[:8]}"
        self.gateway: subprocess.Popen | None = None
        self.panes: list[str] = []

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "TMUX", "API_SERVER_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
        env.update(HOME=str(self.home), HERMES_HOME=str(self.hermes_home), PYTHONPATH=str(REPO_ROOT),
                   HERMES_GATEWAY_LOCK_DIR=str(self.root / "locks"), TERM="xterm-256color",
                   HERMES_STATE_DB_GUARD_BYPASS="1", PYTHONDONTWRITEBYTECODE="1")
        return env

    # gateway -----------------------------------------------------------------------------
    def start_gateway(self) -> None:
        (self.root / "locks").mkdir(exist_ok=True)
        log = open(self.root / "gateway.log", "wb")  # noqa: SIM115 - closed in stop_gateway
        self.gateway = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=self.project, env=self.env(),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        self.gateway._log = log  # type: ignore[attr-defined]
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            assert self.gateway.poll() is None, self.gateway_log()
            with contextlib.suppress(Exception):
                status, body = self.http("GET", "/health/detailed")
                if status == 200 and json.loads(body).get("pid") == self.gateway.pid:
                    return
            time.sleep(0.3)
        raise AssertionError("gateway never became ready\n" + self.gateway_log())

    def stop_gateway(self) -> None:
        proc, self.gateway = self.gateway, None
        if proc is None:
            return
        if proc.poll() is None:
            subprocess.run([sys.executable, "-c",
                            "import sys; from gateway.status import write_planned_stop_marker as w; "
                            "w(int(sys.argv[1]))", str(proc.pid)],
                           cwd=self.project, env=self.env(), capture_output=True, timeout=60)
            time.sleep(0.3)
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        proc._log.close()  # type: ignore[attr-defined]

    def gateway_log(self) -> str:
        with contextlib.suppress(OSError):
            return (self.root / "gateway.log").read_text(errors="replace")[-4000:]
        return ""

    def http(self, method: str, path: str) -> tuple[int, str]:
        req = urllib.request.Request(self.base + path, method=method,
                                     headers={"Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    def activities(self) -> list[dict[str, Any]]:
        status, body = self.http("GET", "/v1/activity")
        assert status == 200, f"/v1/activity -> {status}: {body[:500]}"
        return json.loads(body)["activities"]

    def wait_activity(self, predicate: Callable[[list[dict]], Any], what: str, timeout: float = 90) -> Any:
        deadline = time.monotonic() + timeout
        last: list[dict] = []
        while time.monotonic() < deadline:
            last = self.activities()
            hit = predicate(last)
            if hit:
                return hit
            time.sleep(0.1)
        raise AssertionError(f"{what} never seen; last={json.dumps(last, ensure_ascii=False)[:2000]}")

    # tmux CLI ----------------------------------------------------------------------------
    def tmux(self, *args: str) -> str:
        # Targets are written ``=<name>:``: tmux resolves a bare short session name fuzzily.
        return subprocess.run(["tmux", "-L", self.sock, *args], capture_output=True, text=True, timeout=30).stdout

    def start_cli(self, name: str) -> None:
        argv = [sys.executable, "-m", "hermes_cli.main", "chat", "--cli"]
        if not self.panes:
            subprocess.run(["tmux", "-L", self.sock, "-f", os.devnull, "new-session", "-d", "-s", name,
                            "-x", "140", "-y", "40", "-c", str(self.project), *argv],
                           env=self.env(), check=True, timeout=30)
        else:
            self.tmux("new-session", "-d", "-s", name, "-x", "140", "-y", "40", "-c", str(self.project), *argv)
        self.panes.append(name)
        self.wait_screen(name, "Welcome to Hermes", timeout=150)
        time.sleep(1.5)

    def screen(self, name: str) -> str:
        return self.tmux("capture-pane", "-p", "-J", "-t", f"={name}:", "-S", "-", "-E", "-")

    def wait_screen(self, name: str, needle: str, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        while needle not in self.screen(name):
            assert time.monotonic() < end, f"{needle!r} never appeared:\n{self.screen(name)[-3000:]}"
            time.sleep(0.1)

    def ask(self, name: str, text: str) -> None:
        self.tmux("send-keys", "-t", f"={name}:", "-l", text)
        time.sleep(0.5)  # typed text + Enter in one write is a paste, not a submit
        self.tmux("send-keys", "-t", f"={name}:", "Enter")

    def keys(self, name: str, *keys: str) -> None:
        self.tmux("send-keys", "-t", f"={name}:", *keys)

    def pane_pid(self, name: str) -> int:
        return int(self.tmux("display-message", "-p", "-t", f"={name}:", "#{pane_pid}").strip())

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.tmux("kill-server")
        self.stop_gateway()


@pytest.fixture
def harness(tmp_path: Path):
    script = Scripted()
    with FakeLLMServer(script, aux=lambda _r: Text("Scripted title")) as llm:
        h = Harness(tmp_path, llm)
        h.script = script  # type: ignore[attr-defined]
        try:
            h.start_gateway()
            yield h
        finally:
            h.close()


def _only(state: str | set[str]) -> Callable[[list[dict]], Any]:
    states = {state} if isinstance(state, str) else state
    return lambda acts: next((a for a in acts if a.get("surface") == "cli" and a.get("state") in states), None)


def _cli(states: str | set[str], *, session: str | None = None, run_id: str | None = None):
    states_set = {states} if isinstance(states, str) else states
    def pick(acts: list[dict]) -> Any:
        return next((a for a in acts if a.get("surface") == "cli" and a.get("state") in states_set
                     and (session is None or a.get("session_id") == session)
                     and (run_id is None or a.get("run_id") == run_id)), None)
    return pick


class SSE:
    """Background reader of /v1/activity/events recording every parsed event."""

    def __init__(self, h: Harness) -> None:
        req = urllib.request.Request(h.base + "/v1/activity/events", headers={"Authorization": f"Bearer {h.key}"})
        self._resp = urllib.request.urlopen(req, timeout=60)
        self.events: list[dict] = []
        self.closed = threading.Event()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            for raw in self._resp:
                line = raw.decode().rstrip("\n")
                if line.startswith("data: "):
                    self.events.append(json.loads(line[6:]))
        except Exception:
            pass
        finally:
            self.closed.set()

    def states(self, run_id: str) -> list[str]:
        out: list[str] = []
        for ev in self.events:
            acts = ev.get("activities") if ev["event"] == "activity.snapshot" else [ev.get("activity") or {}]
            for a in acts:
                if a.get("run_id") == run_id and (not out or out[-1] != a["state"]):
                    out.append(a["state"])
        return out

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._resp.close()


def test_cli_turn_started_without_any_api_client_message_is_reported_as_thinking(harness: Harness) -> None:
    """RED baseline: nothing reaches /v1/activity unless the CLI publishes its own lifecycle."""
    gate = threading.Event()
    harness.script.push("zz-think", Text("hello from the cli turn"), gate=gate)  # type: ignore[attr-defined]
    harness.start_cli("a")
    harness.ask("a", "zz-think please")
    act = harness.wait_activity(_cli("thinking"), "cli thinking")
    assert act["profile"] and act["session_id"] and act["run_id"] and act["source_alive"] is True
    gate.set()
    harness.wait_activity(_cli("completed", run_id=act["run_id"]), "cli completed")


def test_tool_then_streamed_answer_then_completion_flow_through_sse_in_order(harness: Harness) -> None:
    s = harness.script  # type: ignore[attr-defined]
    tool_gate, answer_gate = threading.Event(), threading.Event()
    s.push("zz-tool", ToolCall("terminal", {"command": "sleep 3"}))
    s.push("zz-tool", Text("tool finished, " + "streaming words " * 40, chunk_chars=8, delay_per_chunk=0.05),
           gate=answer_gate)
    harness.start_cli("a")
    sse = SSE(harness)
    try:
        harness.ask("a", "zz-tool please")
        working = harness.wait_activity(_cli("working"), "cli working")
        assert working["tool"] == "terminal"
        harness.wait_activity(_cli("thinking", run_id=working["run_id"]), "thinking after the tool")
        answer_gate.set()
        harness.wait_activity(_cli("talking", run_id=working["run_id"]), "cli talking")
        done = harness.wait_activity(_cli("completed", run_id=working["run_id"]), "cli completed")
        assert done["error"] is None
        deadline = time.monotonic() + 10
        while "completed" not in sse.states(working["run_id"]) and time.monotonic() < deadline:
            time.sleep(0.1)
        states = sse.states(working["run_id"])
        assert states[:1] == ["thinking"] and "working" in states and "talking" in states, states
        assert states[-1] == "completed", states
        assert states.index("working") < states.index("talking") < states.index("completed"), states
        seqs = [e["activity"]["seq"] for e in sse.events if e["event"] == "activity.updated"
                and e["activity"]["run_id"] == working["run_id"]]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
        assert sse.events[0]["event"] == "activity.snapshot"
    finally:
        sse.close()


def test_approval_prompt_is_waiting_and_only_the_cli_answers_it(harness: Harness) -> None:
    victim = harness.project / "victim"
    victim.mkdir()
    s = harness.script  # type: ignore[attr-defined]
    s.push("zz-approve", ToolCall("terminal", {"command": f"rm -rf {victim}"}))
    s.push("zz-approve", Text("ok, not deleting"))
    harness.start_cli("a")
    harness.ask("a", "zz-approve please")
    waiting = harness.wait_activity(_cli("waiting"), "cli waiting for approval")
    assert waiting["waiting_for"] == "approval"
    # The feed is read-only: there is no route that can answer it.
    status, _ = harness.http("POST", f"/v1/runs/{waiting['run_id']}/approval")
    assert status in (400, 401, 404, 405), status
    time.sleep(1.0)
    assert harness.activities()[0]["state"] == "waiting"
    harness.keys("a", "Down", "Down", "Down")
    time.sleep(0.3)
    harness.keys("a", "Enter")
    harness.wait_activity(_cli("completed", run_id=waiting["run_id"]), "completed after deny")
    assert victim.exists()


def test_ctrl_c_mid_turn_is_cancelled_not_completed(harness: Harness) -> None:
    harness.script.push("zz-cancel", Hang(60))  # type: ignore[attr-defined]
    harness.start_cli("a")
    harness.ask("a", "zz-cancel please")
    act = harness.wait_activity(_cli("thinking"), "cli thinking")
    time.sleep(1.0)
    harness.keys("a", "C-c")
    harness.wait_activity(_cli("cancelled", run_id=act["run_id"]), "cli cancelled")


def test_provider_failure_is_reported_as_failed(harness: Harness) -> None:
    from tests.fakes.fake_llm_provider import Error
    for _ in range(4):
        harness.script.push("zz-fail", Error(400, "scripted bad request"))  # type: ignore[attr-defined]
    harness.start_cli("a")
    harness.ask("a", "zz-fail please")
    failed = harness.wait_activity(_cli("failed"), "cli failed", timeout=120)
    assert failed["error"]


def test_parallel_cli_turns_one_finishing_does_not_end_the_other(harness: Harness) -> None:
    s = harness.script  # type: ignore[attr-defined]
    gate_a, gate_b = threading.Event(), threading.Event()
    s.push("zz-par-a", Text("answer a"), gate=gate_a)
    s.push("zz-par-b", Text("answer b"), gate=gate_b)
    harness.start_cli("a")
    harness.start_cli("b")
    harness.ask("a", "zz-par-a please")
    harness.ask("b", "zz-par-b please")
    both = harness.wait_activity(
        lambda acts: acts if len([a for a in acts if a["state"] == "thinking"]) == 2 else None, "two thinking")
    assert len({a["session_id"] for a in both}) == 2 and len({a["publisher_id"] for a in both}) == 2
    run_a = next(a for a in both if a["started_at"] == min(x["started_at"] for x in both))["run_id"]
    run_b = next(a["run_id"] for a in both if a["run_id"] != run_a)
    gate_b.set()
    harness.wait_activity(_cli("completed", run_id=run_b), "b completed")
    assert _cli("thinking", run_id=run_a)(harness.activities()), harness.activities()
    gate_a.set()
    harness.wait_activity(_cli("completed", run_id=run_a), "a completed")


@pytest.mark.live_system_guard_bypass  # the victim is this test's own tmux pane (checked below)
def test_cli_killed_mid_turn_is_lost_never_idle(harness: Harness) -> None:
    harness.script.push("zz-kill", Hang(60))  # type: ignore[attr-defined]
    harness.start_cli("keep")  # keeps the tmux server alive after the victim pane dies
    harness.start_cli("a")
    harness.ask("a", "zz-kill please")
    act = harness.wait_activity(_cli("thinking"), "cli thinking")
    import psutil
    victim = psutil.Process(harness.pane_pid("a"))
    assert Path(victim.cwd()).resolve() == harness.project.resolve(), victim.cmdline()
    assert "hermes_cli.main" in " ".join(victim.cmdline()), victim.cmdline()
    victim.send_signal(signal.SIGKILL)
    lost = harness.wait_activity(_cli("lost", run_id=act["run_id"]), "cli lost")
    assert lost["source_alive"] is False


def test_gateway_restart_recovers_the_live_cli_activity(harness: Harness) -> None:
    gate = threading.Event()
    harness.script.push("zz-restart", Text("after restart"), gate=gate)  # type: ignore[attr-defined]
    harness.start_cli("a")
    harness.ask("a", "zz-restart please")
    act = harness.wait_activity(_cli("thinking"), "cli thinking")
    sse = SSE(harness)
    harness.stop_gateway()
    assert sse.closed.wait(30), "client stream must end when the gateway goes away"
    harness.start_gateway()
    sse2 = SSE(harness)
    try:
        deadline = time.monotonic() + 10
        while not sse2.events and time.monotonic() < deadline:
            time.sleep(0.05)
        snap = sse2.events[0]
        assert snap["event"] == "activity.snapshot"
        assert [a["state"] for a in snap["activities"] if a["run_id"] == act["run_id"]] == ["thinking"]
        gate.set()
        harness.wait_activity(_cli("completed", run_id=act["run_id"]), "completed after restart")
    finally:
        sse.close()
        sse2.close()
