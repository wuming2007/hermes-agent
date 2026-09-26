"""Behavior contract: which claims in an answer the current turn's tool results support.

The evidence loop must tell a write from a working feature, a test run from a deployment, a
subagent's report from our own check, and a timed-out or still-running command from a result.
Expectations are hand-written literals, never derived from the module under test.
"""

import json

import pytest

from agent.cognition_evidence import build_evidence_report, classify_turn_evidence


def _turn(*steps):
    messages = [{"role": "user", "content": "do the work"}]
    for i, (name, args, result) in enumerate(steps):
        call_id = f"call_{i}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}},
        ]})
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                         "content": result if isinstance(result, str) else json.dumps(result)})
    return messages


WRITE_OK = ("write_file", {"path": "app/config.yaml", "content": "x"}, {"bytes_written": 1, "path": "app/config.yaml"})
WRITE_BLOCKED = ("write_file", {"path": "app/config.yaml", "content": "x"},
                 {"error": "BLOCKED: Tool 'write_file' requires approval but single-query mode denies it."})
TEST_OK = ("terminal", {"command": "python -m pytest tests/unit -q"}, {"output": "5 passed", "exit_code": 0, "error": None})
TEST_FAIL = ("terminal", {"command": "npm test"}, {"output": "1 failed", "exit_code": 1, "error": None})
RUN_OK = ("terminal", {"command": "python app.py --check"}, {"output": "ok", "exit_code": 0, "error": None})
DEPLOY_OK = ("terminal", {"command": "launchctl kickstart -k gui/501/com.example.app"},
             {"output": "", "exit_code": 0, "error": None})
CMD_TIMEOUT = ("terminal", {"command": "make build"}, {"output": "", "exit_code": 124, "error": None})
CMD_RERUN_OK = ("terminal", {"command": "make build"}, {"output": "built", "exit_code": 0, "error": None})
BACKGROUND = ("terminal", {"command": "./deploy.sh", "background": True},
              {"output": "Background process started", "session_id": "proc_1", "pid": 42, "exit_code": 0, "error": None})
YIELDED = ("terminal", {"command": "./long.sh"},
           {"output": "", "exit_code": None, "error": None, "status": "yielded_to_background", "session_id": "proc_2"})
TOOL_TIMEOUT = ("web_extract", {"urls": ["https://example.com/a"]}, "Error executing tool 'web_extract': timed out after 60.0s")
READ_OK = ("read_file", {"path": "app/config.yaml"}, {"content": "x", "total_lines": 1})
DELEGATE = ("delegate_task", {"goal": "fix the parser bug"},
            {"results": [{"status": "completed", "summary": "Fixed it and all tests pass."}]})
BRIDGED_TEST = ("tool_call", {"calls": [{"name": "terminal", "arguments": {"command": "pytest -q"}}]},
                {"output": "3 passed", "exit_code": 0, "error": None})
SHELL_WRITE = ("terminal", {"command": "sed -i 's/debug: true/debug: false/' app/config.yaml"},
               {"output": "", "exit_code": 0, "error": None})
SHELL_READ = ("terminal", {"command": "cat app/config.yaml"}, {"output": "debug: false", "exit_code": 0, "error": None})
PROC_DONE = ("process_manage", {"action": "wait", "session_id": "proc_1"},
             {"status": "exited", "command": "./deploy.sh", "exit_code": 0, "completion_reason": "exited", "output": "ok"})
WEB_PARTIAL = ("web_extract", {"urls": ["https://example.com/a", "https://example.com/b"]},
               {"results": [{"url": "https://example.com/a", "content": "x"},
                            {"url": "https://example.com/b", "content": "", "error": "HTTP 404"}]})


@pytest.mark.parametrize(
    "step, kind, status, provenance",
    [
        pytest.param(WRITE_OK, "write", "ok", "self", id="write-landed"),
        pytest.param(WRITE_BLOCKED, "write", "blocked", "self", id="write-blocked-by-approval"),
        pytest.param(TEST_OK, "test", "ok", "self", id="test-pass"),
        pytest.param(TEST_FAIL, "test", "error", "self", id="test-fail"),
        pytest.param(RUN_OK, "exec", "ok", "self", id="plain-command"),
        pytest.param(DEPLOY_OK, "deploy", "ok", "self", id="service-restart"),
        pytest.param(CMD_TIMEOUT, "exec", "timeout", "self", id="command-timeout-124"),
        pytest.param(BACKGROUND, "background_spawn", "running", "self", id="background-spawn-is-not-a-result"),
        pytest.param(YIELDED, "exec", "running", "self", id="yielded-to-background"),
        pytest.param(TOOL_TIMEOUT, "network", "timeout", "self", id="tool-deadline"),
        pytest.param(READ_OK, "read", "ok", "self", id="read"),
        pytest.param(DELEGATE, "delegate_report", "ok", "subagent", id="subagent-report"),
        pytest.param(BRIDGED_TEST, "test", "ok", "self", id="bridged-test"),
        pytest.param(SHELL_WRITE, "write", "ok", "self", id="shell-edit-is-a-write"),
        pytest.param(SHELL_READ, "read", "ok", "self", id="shell-read-is-a-read"),
        pytest.param(PROC_DONE, "deploy", "ok", "self", id="waited-background-job"),
        pytest.param(WEB_PARTIAL, "network", "partial", "self", id="some-urls-failed"),
    ],
)
def test_each_tool_result_is_classified_by_what_it_did_and_how_it_ended(step, kind, status, provenance):
    (item,) = classify_turn_evidence(_turn(step))

    assert (item.kind, item.status, item.provenance) == (kind, status, provenance)
    assert item.evidence_id == "e_call_0"


@pytest.mark.parametrize(
    "steps, answer, unsupported",
    [
        pytest.param([WRITE_OK], "我已寫入設定檔。", [], id="write-claim-with-write"),
        pytest.param([WRITE_OK], "設定已經修好，現在可以正常使用。",
                     [("working", "no_functional_check_after_write")], id="written-is-not-working"),
        pytest.param([WRITE_OK, TEST_OK], "設定已更新，測試全部通過。", [], id="test-after-write"),
        pytest.param([TEST_OK, WRITE_OK], "測試全部通過。",
                     [("tests_pass", "no_test_after_last_write")], id="stale-test-before-write"),
        pytest.param([TEST_FAIL], "測試通過了。", [("tests_pass", "failed")], id="failed-test"),
        pytest.param([CMD_TIMEOUT], "全部完成了。", [("completed", "timeout")], id="timeout-is-not-a-result"),
        pytest.param([YIELDED], "工作已完成。", [("completed", "still_running")], id="yielded-is-not-a-result"),
        pytest.param([BACKGROUND], "已部署完成。", [("deployed", "still_running")], id="spawn-is-not-deployed"),
        pytest.param([DEPLOY_OK], "服務已重啟完成。", [], id="restart-evidenced"),
        pytest.param([WRITE_OK, TEST_OK], "已部署到正式環境。", [("deployed", "no_deploy_evidence")],
                     id="tested-is-not-deployed"),
        pytest.param([WRITE_BLOCKED], "設定檔已寫入。", [("written", "blocked")], id="blocked-write"),
        pytest.param([DELEGATE], "Claude 已完成修改，並驗證所有測試通過。",
                     [("tests_pass", "only_subagent_report")], id="report-stated-as-fact"),
        pytest.param([DELEGATE], "Claude 回報已完成修改，測試都通過。", [], id="report-attributed"),
        pytest.param([BRIDGED_TEST], "測試全部通過。", [], id="bridged-test-counts"),
        pytest.param([WRITE_OK], "目前尚未部署，也還沒跑測試。", [], id="negated-claims-are-not-claims"),
        pytest.param([BACKGROUND, PROC_DONE], "已部署完成。", [], id="waited-until-the-deploy-exited"),
        pytest.param([WRITE_OK, SHELL_READ], "設定已經修好，現在可以正常使用。",
                     [("working", "no_functional_check_after_write")], id="reading-back-is-not-running"),
        pytest.param([WRITE_OK, RUN_OK], "設定已經修好，現在可以正常使用。", [], id="ran-after-write"),
        pytest.param([], "我已經幫你寄出郵件了。", [("completed", "no_action_evidence")], id="action-without-any-tool"),
        pytest.param([DELEGATE], "全部完成了。", [("completed", "only_subagent_report")], id="subagent-done-as-fact"),
        pytest.param([TEST_OK], "需要的話，我可以幫你部署。", [], id="offer-is-not-a-claim"),
        pytest.param([WRITE_OK], "The config is fixed and works now.",
                     [("working", "no_functional_check_after_write")], id="english-working"),
        pytest.param([TEST_FAIL], "Tests are not passing yet.", [], id="english-negated"),
        pytest.param([TEST_FAIL], "測試沒有全部通過。", [], id="cjk-negated"),
        pytest.param([BACKGROUND], "還沒部署完成。", [], id="not-yet-deployed"),
        pytest.param([TEST_OK, TEST_FAIL], "測試全部通過。", [("tests_pass", "failed")], id="latest-run-decides"),
        pytest.param([CMD_TIMEOUT, RUN_OK], "全部完成了。", [("completed", "timeout")], id="one-timeout-spoils-all-done"),
        pytest.param([CMD_TIMEOUT, CMD_RERUN_OK], "全部完成了。", [], id="rerun-after-timeout"),
        pytest.param([DEPLOY_OK], "已部署到正式環境。測試全部通過。", [("tests_pass", "no_test_evidence")],
                     id="each-sentence-is-its-own-claim"),
    ],
)
def test_claims_need_evidence_of_their_own_kind(steps, answer, unsupported):
    report = build_evidence_report(answer, _turn(*steps))

    assert [(c.claim_class, c.reason) for c in report.unsupported] == unsupported


def test_chat_without_tools_or_state_claims_has_nothing_to_check():
    report = build_evidence_report("今天天氣不錯，適合出去走走。", [{"role": "user", "content": "聊聊天"}])

    assert report.evidence == ()
    assert report.claims == ()


def test_digest_names_every_result_and_gap_within_its_budget():
    steps = [TEST_OK] + [RUN_OK] * 40 + [CMD_TIMEOUT]
    report = build_evidence_report("全部完成了。", _turn(*steps))

    assert len(report.digest) <= 2000
    assert "e_call_0" in report.digest and "timeout" in report.digest
    assert "completed" in report.digest
