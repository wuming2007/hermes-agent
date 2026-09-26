"""Behavior contract: the scope gate keeps execution tools inside what the user asked for.

A stopped task stays stopped (even under /yolo), an unrecorded limit or a "ask me first" rule
goes to the human-approval prompt, a forbidden action is blocked, read-only tools are never
touched, and a gate failure leaves the decision to the normal rules.
"""

import os
import stat
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_PLUGIN = "plugins:\n  enabled: [cognition_wm]\n"
_COGNITION = "cognition:\n  enabled: true\n"


def _task_state(**arguments):
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": arguments}]})


def _run_turns(tmp_path, monkeypatch, script, messages, config):
    from hermes_cli.plugins import discover_plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    monkeypatch.chdir(work)
    results = []
    with FakeLLMServer(script) as server:
        home = Path(os.environ["HERMES_HOME"])
        write_hermes_home(home, server.base_url, extra_config=config)
        discover_plugins(force=True)
        db = SessionDB(db_path=home / "state.db")
        for message in messages:
            agent = AIAgent(provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
                            session_db=db, session_id="sess-gate", quiet_mode=True, platform="cli",
                            enabled_toolsets=["file", "terminal", "cognition_wm"], skip_context_files=True,
                            skip_memory=True)
            results.append(agent.run_conversation(message, conversation_history=db.get_messages_as_conversation(
                "sess-gate")))
    return work, results


def _tool_results(result):
    """This turn's tool results (the messages hold the whole conversation)."""
    messages = result["messages"]
    start = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    return [str(m["content"]) for m in messages[start:] if m.get("role") == "tool"]


@pytest.mark.parametrize("plugin_on", [True, False], ids=["working-memory", "without"])
def test_a_prompt_only_request_keeps_hands_off_the_repo(tmp_path, monkeypatch, plugin_on):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "claude-was-run"
    fake = bin_dir / "claude"
    fake.write_text(f"#!/bin/sh\ntouch '{marker}'\necho started\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    script = [ToolCall("terminal", {"command": f"{fake} -p 'refactor the parser'"}),
              ToolCall("write_file", {"path": "parser.py", "content": "print('x')\n"}),
              Text("這是給 Claude 的 prompt：……")]

    work, (result,) = _run_turns(tmp_path, monkeypatch, script, ["只要幫我寫給 Claude 的 prompt，不要改 repo。"],
                                 (_PLUGIN if plugin_on else "") + _COGNITION)

    touched = marker.exists() or (work / "parser.py").exists()
    assert touched is not plugin_on
    if plugin_on:
        assert all("BLOCKED" in r for r in _tool_results(result))


def test_a_cancelled_task_stays_stopped_even_under_yolo(tmp_path, monkeypatch):
    script = [
        _task_state(action="create", title="重構 parser", goal="重構 parser", kind="engineering"), Text("好。"),
        _task_state(action="set_status", status="cancelled", reason="使用者取消"), Text("已取消。"),
        ToolCall("patch", {"path": "parser.py", "old_string": "a", "new_string": "b"}),
        ToolCall("read_file", {"path": "parser.py"}), Text("好的。"),
    ]
    config = _PLUGIN + _COGNITION + 'approvals:\n  mode: "off"\n'
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "parser.py").write_text("a\n", encoding="utf-8")

    work, results = _run_turns(tmp_path, monkeypatch, script, ["幫我重構 parser", "取消這個任務", "好的"], config)

    patched, read = _tool_results(results[-1])
    assert "BLOCKED" in patched and "CANCELLED" in patched.upper()
    assert "BLOCKED" not in read
    assert (work / "parser.py").read_text(encoding="utf-8") == "a\n"


def test_a_failing_gate_leaves_the_decision_to_the_normal_rules(tmp_path, monkeypatch):
    import plugins.cognition_wm.scope_gate as gate

    def broken(*_a, **_k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(gate, "decide", broken)
    script = [ToolCall("write_file", {"path": "notes.md", "content": "x"}), Text("寫好了。")]

    work, _ = _run_turns(tmp_path, monkeypatch, script, ["不要改 repo，只要寫筆記。"], _PLUGIN + _COGNITION)

    assert (work / "notes.md").exists()


# ---------------------------------------------------------------- decisions


def _turn_with_task(tmp_path, monkeypatch, authorization=None, run_id=None):
    import plugins.cognition_wm.provenance as provenance
    from plugins.cognition_wm import runtime

    monkeypatch.chdir(tmp_path)
    turn = runtime.TurnState(session_id="s1", turn_id="t1", owner=True, project_key=runtime.project_key(),
                             lineage_root="s1", platform="cli")
    source = provenance.turn_source(owner=True, session_id="s1", turn_id="t1")
    wm = runtime.store()
    task = wm.create_task(project_key=turn.project_key, lineage_root="s1", origin_session="s1", title="改 parser",
                          goal="改 parser", kind="engineering", source=source)
    wm.set_focus(task.task_id, source=source)
    if authorization:
        wm.add_item(task.task_id, kind="authorization", text=f"{authorization[1]}：{authorization[0]}",
                    state=authorization[1], attrs={"action_class": authorization[0]}, source=source)
    if run_id:
        monkeypatch.setattr(provenance, "RUN_ID", run_id)
    return turn, task


@pytest.mark.parametrize(
    "authorization, run_id, action",
    [
        pytest.param(("code_change", "forbidden"), None, "block", id="forbidden"),
        pytest.param(("code_change", "ask_first"), None, "approve", id="ask-first"),
        pytest.param(("code_change", "granted"), "a-later-run", "approve", id="grant-from-an-earlier-run"),
        pytest.param(("code_change", "granted"), None, None, id="granted-this-run"),
        pytest.param(("deploy_restart", "forbidden"), None, None, id="rule-for-another-action"),
        pytest.param(None, None, None, id="no-rule"),
    ],
)
def test_authorizations_decide_execution_tools(tmp_path, monkeypatch, authorization, run_id, action):
    from plugins.cognition_wm import runtime, scope_gate

    turn, task = _turn_with_task(tmp_path, monkeypatch, authorization, run_id)
    decision = scope_gate.decide("write_file", {"path": "parser.py", "content": "x"}, turn, runtime.settings())

    assert (decision or {}).get("action") == action
    if action == "approve":
        assert decision["rule_key"] == f"cognition:{task.task_id}:code_change"


@pytest.mark.parametrize(
    "tool_name, args, classes",
    [
        pytest.param("read_file", {"path": "a"}, (), id="read"),
        pytest.param("task_state", {"action": "show"}, (), id="own-tool"),
        pytest.param("write_file", {"path": "a", "content": "x"}, ("code_change",), id="write"),
        pytest.param("terminal", {"command": "ls -la"}, (), id="shell-read"),
        pytest.param("terminal", {"command": "git commit -am wip"}, ("repo_write",), id="git-commit"),
        pytest.param("terminal", {"command": "claude -p 'fix it'"}, ("coding_agent_launch",), id="coding-agent"),
        pytest.param("terminal", {"command": "tmux send-keys -t 1 'y' Enter"}, ("coding_agent_launch",), id="tmux-keys"),
        pytest.param("terminal", {"command": "alembic upgrade head"}, ("data_migration",), id="migration"),
        pytest.param("terminal", {"command": "./deploy.sh prod"}, ("deploy_restart",), id="deploy"),
        pytest.param("terminal", {"command": "sed -i s/a/b/ x.py && git push"}, ("repo_write", "code_change"),
                     id="chained"),
        pytest.param("send_message", {"text": "done"}, ("external_message",), id="message"),
        pytest.param("delegate_task", {"goal": "do it"}, ("coding_agent_launch",), id="delegate"),
    ],
)
def test_execution_tools_are_classified_by_what_they_would_do(tool_name, args, classes):
    from plugins.cognition_wm.scope_gate import action_classes

    assert action_classes(tool_name, args) == classes


@pytest.mark.parametrize(
    "content, blocked",
    [
        pytest.param('<working-memory rev="abc">...', True, id="snapshot"),
        pytest.param("remember task t_0123456789ab is paused", True, id="task-id"),
        pytest.param("The user prefers short answers.", False, id="plain-fact"),
    ],
)
def test_working_memory_never_goes_into_long_term_memory(content, blocked):
    from plugins.cognition_wm.scope_gate import on_pre_tool_call

    decision = on_pre_tool_call(tool_name="memory", args={"action": "add", "target": "memory", "content": content},
                                session_id="s-none")

    assert ((decision or {}).get("action") == "block") is blocked
