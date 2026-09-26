"""Behavior contract: which claims in an answer the current turn's tool results support.

The evidence loop must tell a write from a working feature, a test run from a deployment, a
subagent's report from our own check, and a timed-out or still-running command from a result.
Expectations are hand-written literals, never derived from the module under test.
"""

import json

import pytest

from agent.cognition_claims import CLAIM_CLASSES
from agent.cognition_evidence import (
    SUPPORT,
    VERIFIER_EVIDENCE_BUDGET,
    build_evidence_report,
    classify_turn_evidence,
    evidence_for_verifier,
    evidence_metadata,
    evidence_refs_for,
)


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
BG_FAILED = ("terminal", {"command": "./deploy.sh", "background": True},
             {"output": "", "exit_code": -1, "error": "Failed to start background process: no such file"})
TEST_AND_DEPLOY = ("terminal", {"command": "pytest -q && ./deploy.sh"},
                   {"output": "3 passed\ndeployed", "exit_code": 0, "error": None})
ALPHA_TEST = ("terminal", {"command": "pytest tests/test_alpha.py -q"}, {"output": "4 passed", "exit_code": 0, "error": None})
BETA_TEST = ("terminal", {"command": "pytest tests/test_beta.py -q"}, {"output": "3 passed", "exit_code": 0, "error": None})
SKIPPED_TEST = ("terminal", {"command": "true || pytest tests/test_beta.py"}, {"output": "", "exit_code": 0, "error": None})
MASKED_TEST = ("terminal", {"command": "pytest -q || true"}, {"output": "1 failed", "exit_code": 0, "error": None})
PIPED_TEST = ("terminal", {"command": "pytest -q 2>&1 | tail -5"}, {"output": "1 failed", "exit_code": 0, "error": None})
TEST_THEN_EDIT = ("terminal", {"command": "pytest -q && sed -i 's/a/b/' app/config.yaml"},
                  {"output": "3 passed", "exit_code": 0, "error": None})
AGENT_RUN = ("terminal", {"command": "claude -p 'run pytest and fix the failures'"},
             {"output": "All tests pass now.", "exit_code": 0, "error": None})
AGENT_THEN_TEST = ("terminal", {"command": "claude -p 'fix the parser' && pytest -q"},
                   {"output": "fixed\n3 passed", "exit_code": 0, "error": None})
TASK_STATE_SHOW = ("task_state", {"action": "show"}, {"ok": True, "state": "deliverable: 報告 (done, e_1)"})
TODO_WRITE = ("todo_list", {"todos": [{"id": "1", "content": "驗證報告", "status": "completed"}]},
              {"todos": [{"id": "1", "content": "驗證報告", "status": "completed"}]})
OLD_SESSION = ("session_search", {"query": "部署"},
               {"success": True, "results": [
                   {"session_id": "s_old", "when": "September 20, 2026 at 02:03 PM", "source": "cli",
                    "snippet": "已部署到正式環境，服務正常"}]})
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
        pytest.param(BG_FAILED, "background_spawn", "error", "self", id="spawn-that-never-started"),
        pytest.param(AGENT_RUN, "delegate_report", "ok", "subagent", id="coding-agent-launched-from-the-shell"),
        pytest.param(("terminal", {"command": "touch /tmp/pytest-of-ci/flag"}, {"output": "", "exit_code": 0, "error": None}),
                     "write", "ok", "self", id="path-named-like-a-test-runner"),
        pytest.param(("terminal", {"command": "cat /srv/deploy/config.yaml"}, {"output": "x", "exit_code": 0, "error": None}),
                     "read", "ok", "self", id="path-named-like-deploy"),
        pytest.param(("terminal", {"command": "touch /tmp/deployed.flag"}, {"output": "", "exit_code": 0, "error": None}),
                     "write", "ok", "self", id="argument-named-like-deploy"),
        pytest.param(("terminal", {"command": "cd app && bash scripts/deploy.sh prod"}, {"output": "", "exit_code": 0, "error": None}),
                     "deploy", "ok", "self", id="deploy-script"),
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
                     [("tests_pass", "only_subagent_report"), ("written", "only_subagent_report")],
                     id="report-stated-as-fact"),
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
        pytest.param([DEPLOY_OK], "已重新啟動服務，設定檔也已寫入完成。", [("written", "no_write_evidence")],
                     id="evidenced-claim-does-not-hide-its-neighbour"),
        pytest.param([TEST_AND_DEPLOY], "測試通過，已部署完成。", [], id="chained-command-counts-for-each-step"),
        # after `;` the exit status is the deploy's: the tests ran, but their result is not shown
        pytest.param([("terminal", {"command": "pytest -q; ./deploy.sh"}, {"output": "ok", "exit_code": 0, "error": None})],
                     "測試通過，已部署完成。", [("tests_pass", "result_not_shown")], id="semicolon-chain"),
        pytest.param([DELEGATE], "Claude 回報資料庫已經修好。另外，新介面已經部署上線了。",
                     [("deployed", "only_subagent_report")], id="report-does-not-cover-a-new-subject"),
        pytest.param([BG_FAILED], "已部署完成。", [("deployed", "failed")], id="failed-spawn-is-a-failure"),
        pytest.param([DELEGATE], "Claude 回報已完成修改。測試都通過了。", [], id="attribution-carries-over"),
        # a claim about a named target needs evidence about that target
        pytest.param([ALPHA_TEST], "Beta 的測試通過了。", [("tests_pass", "target_not_shown")],
                     id="another-projects-tests"),
        pytest.param([ALPHA_TEST, BETA_TEST], "Beta 的測試通過了。", [], id="the-named-projects-tests"),
        pytest.param([ALPHA_TEST], "測試通過了。", [], id="no-target-named"),
        # the exit status must show the step's own result
        pytest.param([SKIPPED_TEST], "Beta 的測試通過了。", [("tests_pass", "result_not_shown")],
                     id="test-after-or-may-not-have-run"),
        pytest.param([MASKED_TEST], "測試通過了。", [("tests_pass", "result_not_shown")], id="failure-masked-by-or"),
        pytest.param([PIPED_TEST], "測試通過了。", [("tests_pass", "result_not_shown")], id="exit-status-of-the-pipe"),
        # effects inside one command keep their order
        pytest.param([TEST_THEN_EDIT], "設定已更新，測試全部通過。", [("tests_pass", "no_test_after_last_write")],
                     id="test-then-edit-in-one-command"),
        # a coding agent's success is its report, not our check
        pytest.param([AGENT_RUN], "測試全部通過了。", [("tests_pass", "only_subagent_report")],
                     id="coding-agent-report-stated-as-fact"),
        pytest.param([AGENT_RUN], "Claude 回報測試全部通過。", [], id="coding-agent-report-attributed"),
        pytest.param([AGENT_THEN_TEST], "測試通過了。", [], id="own-test-after-the-agent"),
        # the assistant's own bookkeeping is not evidence
        pytest.param([TASK_STATE_SHOW], "報告已經驗證過了。", [("verified", "no_check_evidence")],
                     id="own-task-state-is-not-a-check"),
        pytest.param([TODO_WRITE], "報告已經驗證過了。", [("verified", "no_check_evidence")],
                     id="own-todo-list-is-not-a-check"),
        # an earlier session's record is not the current state
        pytest.param([OLD_SESSION], "已經部署到正式環境了。", [("deployed", "only_historical_record")],
                     id="old-record-stated-as-now"),
        pytest.param([OLD_SESSION], "服務現在正常運作。", [("working", "only_historical_record")],
                     id="old-record-as-working"),
        pytest.param([OLD_SESSION], "根據上次的紀錄，已經部署到正式環境了。", [], id="old-record-cited"),
        pytest.param([OLD_SESSION], "截至上次紀錄，已經部署到正式環境。", [], id="old-record-as-of"),
        pytest.param([TEST_OK, AGENT_RUN], "測試全部通過了。", [("tests_pass", "only_subagent_report")],
                     id="own-test-before-the-agent-changed-things"),
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


def _notice(exit_code):
    return {"role": "user", "content": (
        f"[IMPORTANT: Background process proc_1 completed normally (exit code {exit_code}).\n"
        "Command: ./deploy.sh\nOutput:\nok]")}


@pytest.mark.parametrize("exit_code, unsupported", [
    pytest.param(0, [], id="finished"),
    pytest.param(1, [("deployed", "failed")], id="failed"),
])
def test_background_completion_notice_is_evidence_for_its_job(exit_code, unsupported):
    messages = _turn(BACKGROUND) + [{"role": "assistant", "content": "部署在背景執行中。"}, _notice(exit_code)]

    report = build_evidence_report("已部署完成。", messages)

    assert [(c.claim_class, c.reason) for c in report.unsupported] == unsupported
    assert (report.evidence[-1].kind, report.evidence[-1].target) == ("deploy", "./deploy.sh")


def test_runtime_nudges_do_not_start_a_new_turn():
    messages = _turn(TEST_OK) + [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Your previous tool call was dropped; answer again.", "_dropped_toolcall_nudge": True},
    ]

    assert build_evidence_report("測試全部通過。", messages).unsupported == ()


def test_verifier_evidence_shows_what_the_answer_relies_on_within_budget():
    padding = [f"/srv/projects/workspace/archive/{i:02d}/notes.md" for i in range(12)]
    listing = ("search_files", {"pattern": "*", "target": "files"},
               {"files": [*padding, "work/alpha.txt", "work/beta.txt", *padding]})
    noisy = ("terminal", {"command": "python app.py --check"}, {"output": "x" * 50_000, "exit_code": 0, "error": None})
    text = evidence_for_verifier("資料夾裡有 alpha.txt 和 beta.txt。", _turn(listing, *[noisy] * 30))

    assert "work/alpha.txt" in text and "work/beta.txt" in text
    assert len(text) <= VERIFIER_EVIDENCE_BUDGET


def test_each_sentence_cites_the_evidence_it_was_judged_on():
    answer = "我已寫入設定檔。測試全部通過。已部署到正式環境。設定檔裡的 debug 是 false。"
    report = build_evidence_report(answer, _turn(READ_OK, WRITE_OK, TEST_OK))

    assert evidence_refs_for("我已寫入設定檔。", report) == ("e_call_1",)
    assert evidence_refs_for("測試全部通過。", report) == ("e_call_2",)
    assert evidence_refs_for("已部署到正式環境。", report) == ()
    assert evidence_refs_for("設定檔裡的 debug 是 false。", report) == ("e_call_0", "e_call_2")


def test_a_sentence_keeps_the_attribution_of_the_one_before():
    report = build_evidence_report("Claude 回報已完成修改。測試都通過了。", _turn(DELEGATE))

    assert evidence_refs_for("測試都通過了。", report) == ("e_call_0",)


def test_unsupported_line_shows_the_unbacked_clause_of_a_long_sentence():
    answer = ("我已經修改了整個微服務叢集當中最核心也最複雜的那個設定檔案 app/config.yaml，並且很仔細地重新跑過"
              "一輪完整的單元與整合測試，全部都通過了，服務也已經部署上線了。")
    report = build_evidence_report(answer, _turn(WRITE_OK, TEST_OK))

    (line,) = [line for line in report.digest.splitlines() if line.startswith("- [deployed]")]
    assert "部署上線" in line


def test_evidence_metadata_summarizes_the_report():
    meta = evidence_metadata(build_evidence_report("我已寫入設定檔。測試全部通過。", _turn(TEST_OK, WRITE_OK)))

    assert meta["evidence_tool_result_count"] == 2
    assert (meta["evidence_claim_count"], meta["evidence_supported_count"]) == (2, 1)
    assert meta["evidence_unsupported"] == [
        {"class": "tests_pass", "reason": "no_test_after_last_write", "text": "測試全部通過。"}]
    assert meta["tool_evidence_present"] is True


def test_evidence_from_an_earlier_turn_does_not_count():
    messages = _turn(TEST_OK) + [{"role": "assistant", "content": "測試全部通過。"},
                                 {"role": "user", "content": "再確認一次？"}]

    report = build_evidence_report("測試全部通過。", messages)

    assert [(c.claim_class, c.reason) for c in report.unsupported] == [("tests_pass", "no_test_evidence")]


def test_every_claim_class_has_a_support_rule():
    assert set(CLAIM_CLASSES) == set(SUPPORT)


@pytest.mark.parametrize("status, reason", [("completed", "only_subagent_report"), ("failed", "failed")])
def test_an_async_subagent_result_is_a_subagent_report(status, reason):
    from tools.process_registry_notifications import format_process_notification

    notice = format_process_notification({"type": "async_delegation", "delegation_id": "deleg_1",
                                          "goal": "Fix the parser and run the tests", "status": status,
                                          "summary": "Fixed it; all tests pass.", "api_calls": 3})
    messages = [{"role": "user", "content": "fix the parser"}, {"role": "assistant", "content": "dispatched"},
                {"role": "user", "content": notice}]

    report = build_evidence_report("所有測試都通過了。", messages)

    assert (report.evidence[-1].kind, report.evidence[-1].provenance) == ("delegate_report", "subagent")
    assert [(c.claim_class, c.reason) for c in report.unsupported] == [("tests_pass", reason)]


def test_a_claim_resting_on_an_old_record_carries_the_records_time():
    report = build_evidence_report("已經部署到正式環境了。", _turn(OLD_SESSION))

    (claim,) = report.unsupported
    assert claim.detail == "September 20, 2026 at 02:03 PM"
