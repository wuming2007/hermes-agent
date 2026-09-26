"""Behavior contract: an answer that states results the turn's tools do not show gets one bounded,
read-only check, and whatever stays unverified is marked in the answer that is stored and returned.

Runs the real agent loop against a scripted provider; the evidence loop is opt-in
(``cognition.evidence_loop.enabled``).
"""

import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_CONFIG = """cognition:
  enabled: true
  evidence_loop:
    enabled: true
    max_rounds: 1
    max_extra_tool_calls: 2
"""


def _run(tmp_path, script, question, config=_CONFIG, aux=None):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with FakeLLMServer(script, aux=aux) as server:
        write_hermes_home(Path(os.environ["HERMES_HOME"]), server.base_url, extra_config=config)
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = AIAgent(
            provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
            session_db=db, session_id="sess-gate", quiet_mode=True, platform="cli",
            enabled_toolsets=["file", "terminal"], skip_context_files=True, skip_memory=True,
        )
        result = agent.run_conversation(question, conversation_history=[])
        main = server.main_requests()
        aux_prompts = [str(r["messages"][-1].get("content") or "") for r in server.aux_requests()]
    return agent, result, main, db.get_messages_as_conversation("sess-gate"), aux_prompts


def _nudges(messages):
    return [m for m in messages if m.get("role") == "user" and "[Evidence check" in str(m.get("content"))]


def test_unbacked_deploy_claim_gets_one_read_only_check_then_an_evidence_note(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("release notes", encoding="utf-8")
    flag = tmp_path / "deployed.flag"
    claim = "已部署到正式環境。"
    script = [
        Text(claim),
        ToolCall("read_file", {"path": str(notes)}),
        ToolCall("terminal", {"command": f"touch {flag}"}),
        ToolCall("read_file", {"path": str(notes)}),
        ToolCall("read_file", {"path": str(notes)}),
        Text(claim),
    ]
    agent, result, main, stored, _aux = _run(tmp_path, script, "幫我上線新版。")

    assert max(len(_nudges(body["messages"])) for body in main) == 1
    tool_results = [str(m["content"]) for m in result["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 4
    assert "BLOCKED" not in tool_results[0] and "BLOCKED" not in tool_results[2]
    assert "BLOCKED" in tool_results[1] and "BLOCKED" in tool_results[3]
    assert not flag.exists()
    assert result["final_response"].startswith(claim)
    assert "證據狀態" in result["final_response"] and "本輪沒有部署或重啟" in result["final_response"]
    assert stored[-1]["role"] == "assistant" and stored[-1]["content"] == result["final_response"]
    assert _nudges(stored) == [] and _nudges(result["messages"]) == []
    trace = agent._current_turn_cognition_metadata["cognition_trace"]["evidence"]
    assert (trace["rounds_used"], trace["extra_tool_calls"], trace["downgraded"]) == (1, 2, True)


def test_plain_chat_costs_nothing(tmp_path):
    agent, result, main, _stored, _aux = _run(tmp_path, [Text("今天天氣不錯，適合出去走走。")], "今天適合出門嗎？")

    assert len(main) == 1
    assert result["final_response"] == "今天天氣不錯，適合出去走走。"
    assert agent._current_turn_cognition_metadata["cognition_trace"]["evidence"]["rounds_used"] == 0


def test_loop_off_leaves_the_answer_alone(tmp_path):
    config = "cognition:\n  enabled: true\n"
    _agent, result, main, _stored, _aux = _run(tmp_path, [Text("已部署到正式環境。")], "幫我上線新版。", config=config)

    assert len(main) == 1
    assert result["final_response"] == "已部署到正式環境。"


def test_verifier_evidence_can_be_turned_off(tmp_path):
    config = "cognition:\n  enabled: true\n  consistency_guard:\n    evidence_context: false\n"
    listing = tmp_path / "work"
    listing.mkdir()
    (listing / "alpha.txt").write_text("x", encoding="utf-8")
    script = [ToolCall("search_files", {"pattern": "*.txt", "target": "files", "path": str(listing)}),
              Text("之前那個資料夾裡有 alpha.txt。")]
    _agent, _result, _main, _stored, aux = _run(tmp_path, script, "之前那個資料夾裡有什麼？", config=config)

    (prompt,) = [p for p in aux if "consistency verifier" in p]
    assert "Tool evidence" not in prompt



@pytest.mark.parametrize(
    "tool_name, args, allowed",
    [
        pytest.param("read_file", {"path": "a"}, True, id="read-tool"),
        pytest.param("terminal", {"command": "cat app.log"}, True, id="shell-read"),
        pytest.param("terminal", {"command": "launchctl list | grep app"}, True, id="status-query"),
        pytest.param("terminal", {"command": "pytest -q"}, True, id="run-the-tests"),
        pytest.param("terminal", {"command": "curl -s localhost:8080/health"}, True, id="health-check"),
        pytest.param("process_manage", {"action": "poll", "session_id": "p1"}, True, id="poll-a-job"),
        pytest.param("terminal", {"command": "./deploy.sh"}, False, id="deploy"),
        pytest.param("terminal", {"command": "touch done.flag"}, False, id="shell-write"),
        pytest.param("terminal", {"command": "pytest -q && git push"}, False, id="chained-push"),
        pytest.param("terminal", {"command": "touch /tmp/pytest-of-ci/flag"}, False, id="write-under-a-runner-path"),
        pytest.param("terminal", {"command": "pytest -q > report.txt"}, False, id="test-output-to-a-file"),
        pytest.param("terminal", {"command": "tail -f app.log", "background": True}, False, id="background"),
        pytest.param("terminal", {"command": "python3 app.py --help"}, True, id="running-the-program"),
        pytest.param("terminal", {"command": "bash -c \"claude -p 'fix the tests'\""}, False, id="hidden-agent"),
        pytest.param("terminal", {"command": "echo checking\nclaude -p 'fix it'"}, False, id="agent-on-a-later-line"),
        pytest.param("terminal", {"command": "python3 -c \"open('status.txt','w').write('ok')\""}, False,
                     id="one-liner-write"),
        pytest.param("terminal", {"command": "timeout 60 alembic upgrade head"}, False, id="migration"),
        pytest.param("process_manage", {"action": "kill", "session_id": "p1"}, False, id="kill-a-job"),
        pytest.param("write_file", {"path": "a", "content": "x"}, False, id="write-tool"),
        pytest.param("delegate_task", {"goal": "check it"}, False, id="delegate"),
        pytest.param("send_message", {"text": "done"}, False, id="message"),
    ],
)
def test_check_round_runs_only_read_only_calls(tool_name, args, allowed):
    from agent.cognition_evidence_gate import EvidenceRound

    assert (EvidenceRound(max_calls=5)(tool_name, args) is None) is allowed


def test_check_round_stops_at_its_call_budget():
    from agent.cognition_evidence_gate import EvidenceRound

    policy = EvidenceRound(max_calls=1)

    assert policy("read_file", {"path": "a"}) is None
    assert "BLOCKED" in policy("read_file", {"path": "b"})
    assert policy.used == 1


def test_every_unsupported_reason_has_words_for_the_model_and_the_user():
    from agent.cognition_evidence import UNSUPPORTED_REASONS
    from agent.cognition_evidence_gate import REASON_TEXT

    assert UNSUPPORTED_REASONS <= set(REASON_TEXT)
