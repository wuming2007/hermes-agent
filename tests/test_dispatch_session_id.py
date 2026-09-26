"""Tests that handle_function_call forwards session_id into registry.dispatch."""

import json
from unittest.mock import MagicMock, patch


def _make_registry(captured: dict):
    """Return a mock registry whose dispatch records the kwargs it receives."""
    registry = MagicMock()

    def _dispatch(name, args, **kwargs):
        captured.update(kwargs)
        return json.dumps({"result": "ok"})

    registry.dispatch.side_effect = _dispatch
    return registry


class TestSessionIdForwarding:

    def test_standard_path_forwards_session_id(self):
        """registry.dispatch receives session_id on the normal tool path."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="t1",
                session_id="sess-abc",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("session_id") == "sess-abc"

    def test_execute_code_path_forwards_session_id(self):
        """registry.dispatch receives session_id on the execute_code path."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "execute_code",
                {"code": "print(1)"},
                task_id="t1",
                session_id="sess-xyz",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("session_id") == "sess-xyz"




class TestNestedCallIdentity:
    """A call made from inside another tool call (execute_code's RPC) names no session or turn of
    its own; it belongs to the call running it, so hooks, plugins and approvals see that session."""

    @staticmethod
    def _dispatch_within(outer, **call_kwargs):
        from tools.approval_context import reset_current_observability_context, set_current_observability_context

        captured = {}
        tokens = set_current_observability_context(**outer) if outer else None
        try:
            with patch("model_tools.registry", _make_registry(captured)):
                from model_tools import handle_function_call
                handle_function_call("web_search", {"query": "test"}, task_id="t1", skip_pre_tool_call_hook=True,
                                     **call_kwargs)
        finally:
            if tokens is not None:
                reset_current_observability_context(tokens)
        return captured

    def test_a_nested_call_belongs_to_the_enclosing_call(self):
        captured = self._dispatch_within({"session_id": "sess-outer", "turn_id": "turn-outer", "tool_call_id": "c1"})

        assert captured.get("session_id") == "sess-outer"

    def test_a_call_that_names_its_session_keeps_it(self):
        captured = self._dispatch_within({"session_id": "sess-outer", "turn_id": "turn-outer"}, session_id="sess-own")

        assert captured.get("session_id") == "sess-own"

    def test_outside_any_call_no_session_is_invented(self):
        captured = self._dispatch_within(None)

        assert captured.get("session_id") is None
