"""Behavior contract: the verifier and the process monitor judge an answer by the turn's tool evidence.

Production acceptance showed the deep-route verifier rewriting answers that were read straight off
a tool result, because its prompt carried only the user request and the candidate. These tests run
the real agent loop against a scripted provider whose verifier stand-in behaves like that model:
it keeps a claim only when the evidence in its prompt shows it.
"""

import json
import os
import re
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_REVISED = "我無法確認資料夾裡有哪些檔案。"


def _grounded_verifier(record):
    """Keep the candidate only if every file it names appears in the prompt's evidence section."""
    prompt = str(record["body"]["messages"][-1].get("content") or "")
    if "consistency verifier" not in prompt:
        return Text("Fake title")
    evidence, _, candidate = prompt.partition("Candidate response:")
    named = set(re.findall(r"\w+\.txt", candidate))
    if named and all(name in evidence for name in named):
        return Text('```json\n{"verdict": "ok"}\n```')
    return Text(json.dumps({"verdict": "revise", "issues": ["file names not shown"], "revised_response": _REVISED},
                           ensure_ascii=False))


def _run(tmp_path, script, question):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    workdir = tmp_path / "work"
    workdir.mkdir()
    for name in ("alpha.txt", "beta.txt"):
        (workdir / name).write_text("x", encoding="utf-8")
    with FakeLLMServer([step(workdir) for step in script], aux=_grounded_verifier) as server:
        write_hermes_home(Path(os.environ["HERMES_HOME"]), server.base_url, extra_config="cognition:\n  enabled: true\n")
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = AIAgent(
            provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
            session_db=db, session_id="sess-guard", quiet_mode=True, platform="cli",
            enabled_toolsets=["file"], skip_context_files=True, skip_memory=True,
        )
        result = agent.run_conversation(question, conversation_history=[])
        verifier_prompts = [
            str(r["messages"][-1].get("content") or "") for r in server.aux_requests()
            if "consistency verifier" in str(r["messages"][-1].get("content") or "")
        ]
    stored = [m for m in db.get_messages_as_conversation("sess-guard") if m.get("role") == "assistant"]
    return agent, result, verifier_prompts, stored


def _search(workdir):
    return ToolCall("search_files", {"pattern": "*.txt", "target": "files", "path": str(workdir)})


def test_answer_read_off_a_tool_result_survives_the_verifier(tmp_path):
    answer = "之前那個資料夾裡有 alpha.txt 和 beta.txt 兩個檔案。"
    agent, result, prompts, stored = _run(tmp_path, [_search, lambda _w: Text(answer)], "之前那個資料夾裡有哪些 txt 檔？")

    assert agent._current_turn_cognition_metadata["verification_plan"] == "full"
    (prompt,) = prompts
    assert "alpha.txt" in prompt.partition("Candidate response:")[0]
    assert result["final_response"] == answer
    assert stored[-1]["content"] == answer


def test_answer_that_contradicts_the_evidence_is_still_revised(tmp_path):
    answer = "之前那個資料夾裡有 alpha.txt 和 gamma.txt。"
    _agent, result, _prompts, stored = _run(tmp_path, [_search, lambda _w: Text(answer)], "之前那個資料夾裡有哪些 txt 檔？")

    assert result["final_response"] == _REVISED
    assert stored[-1]["content"] == _REVISED


def test_process_monitor_cites_each_claims_own_evidence(tmp_path):
    answer = "我已新增 gamma.txt。測試全部通過。"
    agent, _result, _prompts, stored = _run(
        tmp_path,
        [_search, lambda w: ToolCall("write_file", {"path": str(w / "gamma.txt"), "content": "x"}), lambda _w: Text(answer)],
        "之前說的 gamma.txt 請幫我建立。",
    )
    write_call = next(tc["id"] for m in stored for tc in m.get("tool_calls") or []
                      if tc["function"]["name"] == "write_file")
    claims = {c["text"]: c for c in agent._current_turn_cognition_metadata["process_monitor_claims"]}

    assert claims["我已新增 gamma.txt。"]["evidence_refs"] == [f"e_{write_call}"]
    assert claims["我已新增 gamma.txt。"]["supported"] is True
    assert claims["測試全部通過。"]["evidence_refs"] == []
    assert claims["測試全部通過。"]["supported"] is False
    trace = agent._current_turn_cognition_metadata["cognition_trace"]["evidence"]
    assert trace["unsupported"] == [{"class": "tests_pass", "reason": "no_test_after_last_write", "text": "測試全部通過。"}]


def test_fenced_or_wrapped_verifier_json_is_read():
    from agent.consistency_guard import run_full_consistency_check

    revise = {"verdict": "revise", "issues": ["x"], "revised_response": "fixed"}
    for reply in (f"```json\n{json.dumps(revise)}\n```", f"Here is my verdict: {json.dumps(revise)}"):
        result = run_full_consistency_check(candidate_response="draft", user_message="q", verifier=lambda _p, r=reply: r)
        assert (result.changed, result.final_response) == (True, "fixed")


def test_guard_notes_are_not_tool_evidence():
    from agent.autonomy_telemetry import build_autonomy_telemetry_from_metadata

    telemetry = build_autonomy_telemetry_from_metadata({"mode": "deep", "verification_notes": ["claim X is unsupported"]})

    assert telemetry.evidence_present is False


def test_the_verifier_sees_the_task_state_as_bookkeeping_not_evidence(tmp_path, monkeypatch):
    """Working memory keeps a one-line pointer in the model's context once its snapshot is there;
    the verifier, which sees no history, still gets the whole task state, framed as not evidence."""
    from hermes_cli.plugins import discover_plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    def task_state(**arguments):
        return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": arguments}]})

    workdir = tmp_path / "work"
    workdir.mkdir()
    for name in ("alpha.txt", "beta.txt"):
        (workdir / name).write_text("x", encoding="utf-8")
    monkeypatch.chdir(workdir)
    answer = "之前那個資料夾裡有 alpha.txt 和 beta.txt 兩個檔案。"
    script = [
        task_state(action="create", title="整理資料夾", goal="列出資料夾裡的 txt 檔", kind="research",
                   items=[{"kind": "decision", "text": "只列 txt，不動檔案"}]), Text("好。"),
        _search(workdir), Text(answer),
        _search(workdir), Text(answer),
    ]
    config = "plugins:\n  enabled: [cognition_wm]\ncognition:\n  enabled: true\n"
    with FakeLLMServer(script, aux=_grounded_verifier) as server:
        write_hermes_home(Path(os.environ["HERMES_HOME"]), server.base_url, extra_config=config)
        discover_plugins(force=True)
        db = SessionDB(db_path=tmp_path / "state.db")
        for question in ("幫我整理資料夾", "之前那個資料夾裡有哪些 txt 檔？", "再確認一次之前那個資料夾裡有哪些 txt 檔？"):
            agent = AIAgent(provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
                            session_db=db, session_id="sess-wm-guard", quiet_mode=True, platform="cli",
                            enabled_toolsets=["file", "cognition_wm"], skip_context_files=True, skip_memory=True)
            agent.run_conversation(question, conversation_history=db.get_messages_as_conversation("sess-wm-guard"))
        last_user = [str(m.get("content") or "") for m in server.main_requests()[-1]["messages"]
                     if m.get("role") == "user"][-1]
        prompts = [str(r["messages"][-1].get("content") or "") for r in server.aux_requests()
                   if "consistency verifier" in str(r["messages"][-1].get("content") or "")]

    assert "<working-memory unchanged" in last_user  # the model sees only the pointer on the last turn
    task_block = prompts[-1].partition("Candidate response:")[0]
    assert "整理資料夾" in task_block and "只列 txt，不動檔案" in task_block
    assert "not evidence" in task_block
