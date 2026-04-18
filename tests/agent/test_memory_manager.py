"""Tests for PR2 layered retrieval orchestration in MemoryManager and the
backward-compatible ``MemoryProvider.prefetch_layered`` hook."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.memory_manager import MemoryManager
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


class _LayeredAwareProvider(MemoryProvider):
    """Implements ``prefetch_layered`` natively and records what it received."""

    def __init__(self, name: str = "layered", payload: str = "layered-data"):
        self._name = name
        self._payload = payload
        self.layered_calls: list[tuple[str, tuple[str, ...], str]] = []
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
        return ""

    def prefetch_layered(
        self, query: str, *, layers, session_id: str = ""
    ) -> str:
        self.layered_calls.append((query, tuple(layers), session_id))
        return self._payload


class _RaisingProvider(MemoryProvider):
    """Raises on every layered call to verify non-fatal handling."""

    @property
    def name(self) -> str:
        return "boom"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        raise RuntimeError("legacy prefetch broken")

    def prefetch_layered(self, query: str, *, layers, session_id: str = "") -> str:
        raise RuntimeError("layered prefetch broken")


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


# ---------------------------------------------------------------------------
# Task 3: MemoryManager.prefetch_for_policy orchestration
# ---------------------------------------------------------------------------


class TestManagerPrefetchForPolicy:
    def test_uses_layered_path_when_provider_overrides(self):
        provider = _LayeredAwareProvider(payload="layered-out")
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_for_policy(
            "ping", layers=("principles", "semantic"), session_id="sess"
        )
        assert result == "layered-out"
        assert provider.layered_calls == [("ping", ("principles", "semantic"), "sess")]
        # Native layered impl should NOT have called legacy prefetch.
        assert provider.prefetch_calls == []

    def test_legacy_provider_falls_back_to_prefetch(self):
        provider = _LegacyOnlyProvider(payload="legacy-out")
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_for_policy(
            "ping", layers=("principles",), session_id="sess"
        )
        assert result == "legacy-out"
        assert provider.prefetch_calls == [("ping", "sess")]

    def test_provider_exception_is_non_fatal(self):
        boom = _RaisingProvider()
        good = _LegacyOnlyProvider(name="good", payload="still-here")
        mgr = MemoryManager()
        mgr.add_provider(good)
        # _RaisingProvider has name "boom" (non-builtin), but `good` already
        # took the external slot; manager should reject the second external
        # silently. To exercise non-fatal handling we instead bypass
        # registration and stuff it directly into the providers list.
        mgr._providers.append(boom)

        result = mgr.prefetch_for_policy(
            "ping", layers=("principles",), session_id="sess"
        )
        # Good provider's output survives even though boom raised.
        assert result == "still-here"

    def test_multi_provider_results_concatenated_with_blank_line(self):
        a = _LegacyOnlyProvider(name="a", payload="alpha")
        b = _LayeredAwareProvider(name="b", payload="beta")
        mgr = MemoryManager()
        mgr._providers.append(a)
        mgr._providers.append(b)

        result = mgr.prefetch_for_policy(
            "q", layers=("principles", "semantic"), session_id=""
        )
        assert result == "alpha\n\nbeta"

    def test_empty_layers_still_dispatches_safely(self):
        # The router never emits empty layers, but the API should not crash if
        # a future caller passes them.
        provider = _LayeredAwareProvider(payload="something")
        mgr = MemoryManager()
        mgr.add_provider(provider)
        result = mgr.prefetch_for_policy("q", layers=(), session_id="")
        assert result == "something"
        assert provider.layered_calls == [("q", (), "")]

    def test_blank_provider_output_is_dropped_like_prefetch_all(self):
        provider = _LegacyOnlyProvider(payload="   \n  ")
        mgr = MemoryManager()
        mgr.add_provider(provider)
        result = mgr.prefetch_for_policy("q", layers=("principles",), session_id="")
        assert result == ""
