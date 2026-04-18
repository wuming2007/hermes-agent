"""Tests for the backward-compatible ``MemoryProvider.prefetch_layered`` hook
(PR2 Task 2).

Manager-side ``prefetch_for_policy`` orchestration tests are added in the next
commit alongside the manager implementation."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _LegacyOnlyProvider(MemoryProvider):
    """Implements only the legacy ``prefetch()`` API. Used to verify the
    default ``prefetch_layered`` falls back transparently."""

    def __init__(self, name: str = "legacy", payload: str = ""):
        self._name = name
        self._payload = payload
        self.prefetch_calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self.prefetch_calls.append((query, session_id))
        return self._payload


# ---------------------------------------------------------------------------
# Task 2: backward-compatible default hook on MemoryProvider
# ---------------------------------------------------------------------------


class TestProviderLayeredHookDefault:
    def test_default_layered_falls_back_to_prefetch(self):
        provider = _LegacyOnlyProvider(payload="from-prefetch")
        result = provider.prefetch_layered(
            "what time is it?", layers=("principles", "semantic"), session_id="s1"
        )
        assert result == "from-prefetch"
        # Default impl must call legacy prefetch with the same query / session_id.
        assert provider.prefetch_calls == [("what time is it?", "s1")]

    def test_default_layered_passes_session_id_through(self):
        provider = _LegacyOnlyProvider(payload="x")
        provider.prefetch_layered("q", layers=("principles",), session_id="abc")
        assert provider.prefetch_calls[0][1] == "abc"
