"""Tests for PR2 layered retrieval orchestration in MemoryManager and the
backward-compatible ``MemoryProvider.prefetch_layered`` hook."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.memory_ranker import MemoryCandidate, MemoryObjectMetadata, MemoryRankerConfig
from agent.policy_memory import PolicyMemoryItem


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


class _CandidateProvider(MemoryProvider):
    """Provides structured memory candidates for PR13 ranking tests."""

    def __init__(self, name: str = "candidate", candidates=None):
        self._name = name
        self._candidates = list(candidates or [])
        self.candidate_calls: list[tuple[str, tuple[str, ...], str]] = []
        self.layered_calls: list[tuple[str, tuple[str, ...], str]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def prefetch_candidates(self, query: str, *, layers=(), session_id: str = ""):
        self.candidate_calls.append((query, tuple(layers), session_id))
        return list(self._candidates)

    def prefetch_layered(self, query: str, *, layers, session_id: str = "") -> str:
        self.layered_calls.append((query, tuple(layers), session_id))
        return "legacy-layered"


class _CandidateRaisingProvider(_CandidateProvider):
    def prefetch_candidates(self, query: str, *, layers=(), session_id: str = ""):
        raise RuntimeError("candidate prefetch broken")


class TestManagerPrefetchRankedForPolicy:
    def test_candidate_path_returns_ranked_context(self):
        provider = _CandidateProvider(
            candidates=[
                MemoryCandidate(text="low priority", provider="p", relevance=0.1),
                MemoryCandidate(
                    text="high priority", provider="p", source="s1", relevance=1.0, confidence=1.0
                ),
            ]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy(
            "memory question",
            layers=("semantic",),
            session_id="sess",
            ranker_config=MemoryRankerConfig(max_items=2),
        )

        assert "high priority" in result
        assert result.index("high priority") < result.index("low priority")
        assert "[rank=1" in result
        assert "provider=p" in result
        assert provider.candidate_calls == [("memory question", ("semantic",), "sess")]
        assert provider.layered_calls == []

    def test_no_candidates_falls_back_to_existing_layered_prefetch(self):
        provider = _CandidateProvider(candidates=[])
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy(
            "memory question", layers=("semantic",), session_id="sess"
        )

        assert result == "legacy-layered"
        assert provider.layered_calls == [("memory question", ("semantic",), "sess")]

    def test_candidate_provider_exception_is_non_fatal(self):
        boom = _CandidateRaisingProvider(name="boom")
        good = _CandidateProvider(
            name="good",
            candidates=[MemoryCandidate(text="survives", provider="good", relevance=1.0)],
        )
        mgr = MemoryManager()
        mgr._providers.extend([boom, good])

        result = mgr.prefetch_ranked_for_policy("q", layers=("semantic",), session_id="s")

        assert "survives" in result

    def test_ranker_failure_falls_back_to_existing_layered_prefetch(self, monkeypatch):
        provider = _CandidateProvider(
            candidates=[MemoryCandidate(text="candidate", provider="p", relevance=1.0)]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        def _boom(*args, **kwargs):
            raise RuntimeError("ranker exploded")

        monkeypatch.setattr("agent.memory_manager.rank_memory_candidates", _boom)

        result = mgr.prefetch_ranked_for_policy("q", layers=("semantic",), session_id="s")

        assert result == "legacy-layered"


class _MetadataDescribingProvider(_CandidateProvider):
    def __init__(self, name: str = "meta", candidates=None, description=None):
        super().__init__(name=name, candidates=candidates)
        self._description = description or {"keys": ["source_trace", "confidence"]}

    def describe_memory_object_metadata(self):
        return dict(self._description)


class _MetadataDescriptionRaisingProvider(_CandidateProvider):
    def describe_memory_object_metadata(self):
        raise RuntimeError("description broken")


class TestMemoryObjectMetadataManager:
    def test_ranked_prefetch_normalizes_candidate_metadata_into_context(self):
        provider = _CandidateProvider(
            candidates=[
                MemoryCandidate(
                    text="auditable memory",
                    provider="p",
                    relevance=1.0,
                    metadata={
                        "source_trace": ["MEMORY.md", "USER.md"],
                        "confidence": 0.7,
                        "status": "active",
                        "last_verified_at": "2026-04-27",
                    },
                )
            ]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy("q", layers=("semantic",), session_id="s")

        assert "auditable memory" in result
        assert "source_trace=MEMORY.md>USER.md" in result
        assert "confidence=0.70" in result
        assert "status=active" in result
        assert "verified=2026-04-27" in result

    def test_ranked_prefetch_normalizes_candidate_metadata_object(self):
        provider = _CandidateProvider(
            candidates=[
                MemoryCandidate(
                    text="metadata object",
                    provider="p",
                    relevance=1.0,
                    metadata=MemoryObjectMetadata(
                        source_trace=("provider-db",), confidence=0.9, status="inferred"
                    ),
                )
            ]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy("q", layers=("semantic",), session_id="s")

        assert "metadata object" in result
        assert "source_trace=provider-db" in result
        assert "confidence=0.90" in result
        assert "status=inferred" in result

    def test_describe_memory_metadata_support_collects_provider_descriptions(self):
        provider = _MetadataDescribingProvider(
            name="meta-provider",
            description={"supports": ["source_trace", "confidence"], "version": 1},
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.describe_memory_metadata_support()

        assert result == {
            "providers": {
                "meta-provider": {"supports": ["source_trace", "confidence"], "version": 1}
            }
        }

    def test_describe_memory_metadata_support_exception_is_non_fatal(self):
        boom = _MetadataDescriptionRaisingProvider(name="boom")
        good = _MetadataDescribingProvider(name="good", description={"version": 1})
        mgr = MemoryManager()
        mgr._providers.extend([boom, good])

        result = mgr.describe_memory_metadata_support()

        assert result == {"providers": {"good": {"version": 1}}}


class _PolicyProvider(_CandidateProvider):
    def __init__(self, name: str = "policy-provider", candidates=None, policy_items=None):
        super().__init__(name=name, candidates=candidates)
        self._policy_items = list(policy_items or [])
        self.policy_calls: list[tuple[str, str]] = []

    def prefetch_policy_items(self, query: str, *, session_id: str = ""):
        self.policy_calls.append((query, session_id))
        return list(self._policy_items)


class _PolicyRaisingProvider(_CandidateProvider):
    def prefetch_policy_items(self, query: str, *, session_id: str = ""):
        raise RuntimeError("policy broken")


class TestPolicyMemoryManager:
    def test_prefetch_policy_for_query_returns_context_and_metadata(self):
        provider = _PolicyProvider(
            policy_items=[
                PolicyMemoryItem(
                    "email-guard",
                    "Email guard",
                    "Do not send email from scheduled reports.",
                    category="external_action",
                    priority=90,
                    source_trace=("USER.md",),
                    tags=("email", "send"),
                )
            ]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        context, metadata = mgr.prefetch_policy_for_query(
            "send email report", session_id="sess", max_items=5
        )

        assert "policy:email-guard@1" in context
        assert "Do not send email" in context
        assert metadata["count"] == 1
        assert metadata["policy_ids"] == ["email-guard"]
        assert metadata["citations"] == ["policy:email-guard@1"]
        assert metadata["categories"] == ["external_action"]
        assert provider.policy_calls == [("send email report", "sess")]
        assert mgr.last_policy_recall_metadata == metadata

    def test_prefetch_policy_for_query_provider_exception_is_non_fatal(self):
        boom = _PolicyRaisingProvider(name="boom")
        good = _PolicyProvider(
            name="good",
            policy_items=[PolicyMemoryItem("privacy", "Privacy", "Private data stays private", category="privacy")],
        )
        mgr = MemoryManager()
        mgr._providers.extend([boom, good])

        context, metadata = mgr.prefetch_policy_for_query("private data", session_id="s")

        assert "Private data stays private" in context
        assert metadata["policy_ids"] == ["privacy"]

    def test_ranked_prefetch_includes_policy_candidate_when_principles_layer_requested(self):
        provider = _PolicyProvider(
            policy_items=[
                PolicyMemoryItem(
                    "send-guard",
                    "Send guard",
                    "Confirm before sending email.",
                    category="external_action",
                    priority=95,
                    tags=("send", "email"),
                )
            ]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy(
            "send email", layers=("principles", "semantic"), session_id="s"
        )

        assert "Confirm before sending email." in result
        assert "provider=policy" in result
        assert "source=policy:send-guard" in result
        assert "source_trace=policy:send-guard" in result
        assert mgr.last_policy_recall_metadata["policy_ids"] == ["send-guard"]

    def test_ranked_prefetch_skips_policy_when_principles_layer_absent(self):
        provider = _PolicyProvider(
            policy_items=[PolicyMemoryItem("send-guard", "Send guard", "Confirm before sending email.")]
        )
        mgr = MemoryManager()
        mgr.add_provider(provider)

        result = mgr.prefetch_ranked_for_policy(
            "send email", layers=("semantic",), session_id="s"
        )

        assert "Confirm before sending email." not in result
        assert provider.policy_calls == []
        assert mgr.last_policy_recall_metadata["count"] == 0
