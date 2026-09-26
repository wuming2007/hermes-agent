"""Behavior contract: a resumed agent gets back the todo list the model kept earlier.

The gateway builds a fresh AIAgent per message and the CLI rebuilds one on ``--resume``, so the
only durable copy of the todo list is the ``todo_list`` tool result in the transcript. Since the
tool was renamed from ``todo`` to ``todo_list`` and moved behind the ``tool_call`` bridge, the
history pairing check must accept every shape the model can actually use to call it.
"""

import json

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

_TODOS = [{"id": "1", "content": "draft the plan", "status": "in_progress"}]


def _agent(server, db, session_id):
    from run_agent import AIAgent

    return AIAgent(
        provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
        session_db=db, session_id=session_id, quiet_mode=True, platform="cli",
        enabled_toolsets=["todo"], skip_context_files=True, skip_memory=True,
    )


def test_resumed_agent_restores_the_todo_list_the_model_kept(tmp_path):
    """The model keeps its list through the bridge; a fresh agent resuming from state.db has it."""
    from hermes_state import SessionDB

    script = [
        ToolCall("tool_call", {"calls": [{"name": "todo_list", "arguments": {"todos": _TODOS}}]}),
        Text("planned"),
        Text("resumed"),
    ]
    with FakeLLMServer(script) as server:
        db = SessionDB(db_path=tmp_path / "state.db")
        first = _agent(server, db, "sess-todo")
        first.run_conversation("plan the work", conversation_history=[])
        assert first._todo_store.has_items()

        history = db.get_messages_as_conversation("sess-todo")
        resumed = _agent(server, db, "sess-todo")
        resumed.run_conversation("carry on", conversation_history=history)

    assert [t["content"] for t in resumed._todo_store.read()] == ["draft the plan"]


def _history(tool_calls, tool_call_id="c1"):
    result = json.dumps({"todos": _TODOS, "revision": 2, "summary": "1 item"})
    return [
        {"role": "user", "content": "plan the work"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": tool_call_id, "content": result},
        {"role": "assistant", "content": "planned"},
    ]


def _call(name, arguments, call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.parametrize(
    "tool_calls, restored",
    [
        pytest.param([_call("todo_list", {"todos": _TODOS})], 1, id="direct-current-name"),
        pytest.param([_call("todo", {"todos": _TODOS})], 1, id="legacy-alias"),
        pytest.param([_call("tool_call", {"calls": [{"name": "todo_list", "arguments": {"todos": _TODOS}}]})], 1,
                     id="bridge-current-name"),
        pytest.param([_call("tool_call", {"calls": [{"name": "todo", "arguments": {"todos": _TODOS}}]})], 1,
                     id="bridge-legacy-alias"),
        pytest.param([_call("tool_call", {"calls": [{"name": "read_file", "arguments": {"path": "x"}}]})], 0,
                     id="bridge-other-tool"),
        pytest.param([_call("read_file", {"path": "x"})], 0, id="other-tool"),
        pytest.param([_call("todo_list", {"todos": _TODOS}, call_id="other-id")], 0, id="unpaired-result"),
    ],
)
def test_todo_result_counts_only_when_paired_with_a_todo_call(tmp_path, tool_calls, restored):
    from hermes_state import SessionDB

    with FakeLLMServer([]) as server:
        agent = _agent(server, SessionDB(db_path=tmp_path / "state.db"), "sess-shapes")
        agent._hydrate_todo_store(_history(tool_calls))

    assert len(agent._todo_store.read()) == restored
