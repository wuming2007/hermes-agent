"""Behavior contract: a tool whitelist set for a turn governs the tools that turn dispatches.

The background review fork and side questions advertise the full tool schema (prompt-cache parity)
and rely on this dispatch-side whitelist. Tools execute on worker threads, so a whitelist that
lived only on the thread that set it never reached them and every tool ran unrestricted.
"""

from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall


def test_whitelist_blocks_a_tool_the_turn_dispatches_on_a_worker_thread(tmp_path):
    from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist
    from hermes_state import SessionDB
    from run_agent import AIAgent

    target = tmp_path / "must_not_exist.txt"
    script = [ToolCall("write_file", {"path": str(target), "content": "x"}), Text("done")]
    with FakeLLMServer(script) as server:
        agent = AIAgent(
            provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
            session_db=SessionDB(db_path=tmp_path / "state.db"), session_id="sess-wl", quiet_mode=True,
            platform="cli", enabled_toolsets=["file"], skip_context_files=True, skip_memory=True,
        )
        set_thread_tool_whitelist({"read_file"}, deny_msg_fmt="DENIED {tool_name}")
        try:
            result = agent.run_conversation("write it", conversation_history=[])
        finally:
            clear_thread_tool_whitelist()

    (tool_row,) = [m for m in result["messages"] if m.get("role") == "tool"]
    assert not Path(target).exists()
    assert "DENIED write_file" in str(tool_row["content"])
