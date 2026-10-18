#!/usr/bin/env python3
"""委派時的「按子任務選模型」能力（delegation.model_roles）。

這一批測試先以 RED 的形式暴露缺口：目前 ``delegate_task`` 只有全域的
``delegation.provider`` / ``delegation.model`` 覆寫，整批子代理只能共用同一組
憑證。我們要的是「同一批 tasks 裡，每個任務各自選一個已配置的模型角色」，
而且：

* 沒指定角色的任務，行為必須與現在完全一樣（繼承父代理或既有全域覆寫）。
* 指定的角色若不存在／解析失敗，必須明確失敗，不得偷偷換一個模型還宣稱成功。
* 主代理（parent agent）自己的 provider／model 絕對不能被動到。
* 非同步（背景）完成的結果，仍要帶回可辨識的模型身分。

測試一律 mock 掉真實模型連線（patch ``run_agent.AIAgent``），所以不會產生費用。
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.process_registry import process_registry


# ---------------------------------------------------------------------------
# 共用假物件
# ---------------------------------------------------------------------------

def _make_mock_parent(depth=0):
    """建立 delegate_task 需要的最小父代理替身。"""
    parent = MagicMock()
    parent.base_url = "https://api.openai.com/v1"
    parent.api_key = "parent-key"
    parent.provider = "openai"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._interrupt_requested = False
    parent.session_id = "parent-sess"
    return parent


# 典型設定：review 走 OpenAI 路線；second_opinion 是選用的 Gemini 第二意見。
_ROLES_CFG = {
    "max_iterations": 40,
    "model_roles": {
        "review": {
            "provider": "openai",
            "model": "review-model-id",
            "reasoning_effort": "high",
        },
        "second_opinion": {
            "provider": "gemini",
            "model": "second-opinion-model-id",
            "reasoning_effort": "medium",
        },
    },
}


def _fake_resolver(defn, _parent_agent):
    """模擬 _resolve_delegation_credentials：把角色定義變成完整憑證 bundle。

    行為刻意對齊真品：什麼都沒設定時回傳一組全 None（代表「子代理完整繼承
    父代理」），只有真的解析不出來才 raise。
    """
    provider = (defn.get("provider") or "").strip() or None
    model = (defn.get("model") or "").strip() or None
    if provider is None and model is None and not defn.get("base_url"):
        return {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
        }
    if provider == "unavailable":
        raise ValueError("Cannot resolve delegation provider 'unavailable': no API key")
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }
    return {
        "model": model,
        "provider": provider,
        "base_url": base_urls.get(provider, "https://example.invalid/v1"),
        "api_key": f"key-for-{provider}",
        "api_mode": "chat_completions",
        "request_overrides": None,
        "max_output_tokens": None,
    }


def _mock_agent_factory():
    """產生一個每次呼叫都回傳新 child 替身的 AIAgent 替身。"""
    created = []

    def _factory(*_args, **kwargs):
        child = MagicMock()
        child.model = kwargs.get("model") or "parent-model"
        child.provider = kwargs.get("provider") or "openai"
        child.base_url = kwargs.get("base_url")
        child.reasoning_config = kwargs.get("reasoning_config")
        child.session_id = f"child-{len(created)}"
        child.session_prompt_tokens = 10
        child.session_completion_tokens = 5
        child.session_reasoning_tokens = 0
        child._credential_pool = None
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "api_calls": 1,
        }
        child.get_activity_summary.return_value = {
            "api_call_count": 1,
            "current_tool": None,
            "last_activity_ts": time.time(),
        }
        created.append(child)
        return child

    return _factory, created


# ---------------------------------------------------------------------------
# 1. 設定解析（可信任設定 → provider / model / reasoning effort）
# ---------------------------------------------------------------------------

class TestModelRoleConfigParsing:
    def test_valid_roles_are_parsed(self):
        from tools.delegation_model_roles import parse_model_roles

        valid, problems = parse_model_roles(_ROLES_CFG)
        assert set(valid) == {"review", "second_opinion"}
        assert problems == {}
        assert valid["review"]["model"] == "review-model-id"

    def test_reserved_delegation_role_names_are_refused(self):
        """model_role 不可與既有的 leaf／orchestrator 混用。"""
        from tools.delegation_model_roles import parse_model_roles

        valid, problems = parse_model_roles(
            {"model_roles": {"leaf": {"provider": "openai", "model": "x"}}}
        )
        assert "leaf" not in valid
        assert "leaf" in problems

    def test_role_without_any_target_is_a_problem(self):
        from tools.delegation_model_roles import parse_model_roles

        valid, problems = parse_model_roles({"model_roles": {"empty": {}}})
        assert "empty" not in valid
        assert "empty" in problems

    def test_unknown_reasoning_effort_is_a_problem_not_a_silent_default(self):
        from tools.delegation_model_roles import parse_model_roles

        valid, problems = parse_model_roles(
            {
                "model_roles": {
                    "review": {
                        "provider": "openai",
                        "model": "x",
                        "reasoning_effort": "turbo-max-plus",
                    }
                }
            }
        )
        assert "review" not in valid
        assert "turbo-max-plus" in problems["review"]

    def test_available_names_sorted(self):
        from tools.delegation_model_roles import available_model_role_names

        assert available_model_role_names(_ROLES_CFG) == ["review", "second_opinion"]
        assert available_model_role_names({}) == []

    def test_unknown_role_raises_with_configured_names_listed(self):
        from tools.delegation_model_roles import ModelRoleError, resolve_model_role

        with pytest.raises(ModelRoleError) as exc:
            resolve_model_role("nope", _ROLES_CFG, _make_mock_parent(), _fake_resolver)
        msg = str(exc.value)
        assert "nope" in msg
        assert "review" in msg and "second_opinion" in msg

    def test_resolution_failure_surfaces_the_real_reason(self):
        from tools.delegation_model_roles import ModelRoleError, resolve_model_role

        cfg = {"model_roles": {"review": {"provider": "unavailable", "model": "x"}}}
        with pytest.raises(ModelRoleError) as exc:
            resolve_model_role("review", cfg, _make_mock_parent(), _fake_resolver)
        assert "unavailable" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. 核心 RED：同批子任務選不同角色，實際 child 建立路徑要收到各自的憑證
# ---------------------------------------------------------------------------

class TestPerTaskModelRoleReachesChildConstruction:
    def test_distinct_roles_reach_distinct_children(self):
        """一批三個任務：未指定 / review / second_opinion。

        AIAgent（真正的 child 建立路徑）必須收到三組不同的 provider／model，
        而父代理自己完全不變。
        """
        import tools.delegate_tool as dt

        parent = _make_mock_parent(depth=0)
        parent_provider_before = parent.provider
        parent_model_before = parent.model
        factory, created = _mock_agent_factory()

        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = dt.delegate_task(
                tasks=[
                    {"goal": "task-inherit"},
                    {"goal": "task-review", "model_role": "review"},
                    {"goal": "task-second", "model_role": "second_opinion"},
                ],
                parent_agent=parent,
            )

        assert MockAgent.call_count == 3, MockAgent.call_args_list
        kwargs_by_model = {
            c.kwargs.get("model"): c.kwargs for c in MockAgent.call_args_list
        }
        # 未指定角色 → 沿用既有全域行為（這裡 cfg 無 provider/model，所以繼承父代理）
        assert None in kwargs_by_model or "parent-model" in kwargs_by_model
        # review
        assert "review-model-id" in kwargs_by_model
        assert kwargs_by_model["review-model-id"]["provider"] == "openai"
        assert kwargs_by_model["review-model-id"]["api_key"] == "key-for-openai"
        # second_opinion
        assert "second-opinion-model-id" in kwargs_by_model
        assert kwargs_by_model["second-opinion-model-id"]["provider"] == "gemini"
        assert (
            kwargs_by_model["second-opinion-model-id"]["base_url"]
            == "https://generativelanguage.googleapis.com/v1beta/openai"
        )

        # 主代理不受影響
        assert parent.provider == parent_provider_before
        assert parent.model == parent_model_before

        parsed = json.loads(out)
        assert len(parsed["results"]) == 3

    def test_unspecified_role_keeps_legacy_global_override(self):
        """沒有 model_role 的任務，仍走原本的 delegation.provider/model 覆寫。"""
        import tools.delegate_tool as dt

        cfg = dict(_ROLES_CFG, provider="openrouter", model="legacy-global-model")
        legacy_creds = {
            "model": "legacy-global-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-legacy",
            "api_mode": "chat_completions",
        }
        factory, _created = _mock_agent_factory()

        with (
            patch.object(dt, "_load_config", return_value=cfg),
            patch.object(
                dt, "_resolve_delegation_credentials", return_value=legacy_creds
            ),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(goal="plain task", parent_agent=_make_mock_parent())

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["model"] == "legacy-global-model"
        assert kwargs["provider"] == "openrouter"
        assert kwargs["api_key"] == "sk-legacy"

    def test_top_level_model_role_applies_to_single_goal(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(
                goal="review the diff",
                model_role="review",
                parent_agent=_make_mock_parent(),
            )

        assert MockAgent.call_args.kwargs["model"] == "review-model-id"

    def test_per_task_role_beats_top_level(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(
                tasks=[{"goal": "a", "model_role": "second_opinion"}],
                model_role="review",
                parent_agent=_make_mock_parent(),
            )

        assert MockAgent.call_args.kwargs["model"] == "second-opinion-model-id"

    def test_role_reasoning_effort_reaches_child(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(
                goal="review", model_role="review", parent_agent=_make_mock_parent()
            )

        assert MockAgent.call_args.kwargs["reasoning_config"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_role_reasoning_effort_beats_global_delegation_effort(self):
        import tools.delegate_tool as dt

        cfg = dict(_ROLES_CFG, reasoning_effort="low")
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=cfg),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(
                goal="review", model_role="review", parent_agent=_make_mock_parent()
            )

        assert MockAgent.call_args.kwargs["reasoning_config"]["effort"] == "high"


# ---------------------------------------------------------------------------
# 3. 失敗必須明確，不能偷偷 fallback
# ---------------------------------------------------------------------------

class TestModelRoleFailuresAreLoud:
    def test_unknown_role_fails_before_any_child_is_spawned(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = dt.delegate_task(
                tasks=[
                    {"goal": "ok"},
                    {"goal": "bad", "model_role": "ghost_role"},
                ],
                parent_agent=_make_mock_parent(),
            )

        parsed = json.loads(out)
        assert "error" in parsed
        assert "ghost_role" in parsed["error"]
        assert "review" in parsed["error"]  # 列出實際可用的角色
        assert MockAgent.call_count == 0  # 完全沒有子代理被建立

    def test_unresolvable_role_fails_without_substituting_a_model(self):
        import tools.delegate_tool as dt

        cfg = {
            "max_iterations": 40,
            "provider": "openrouter",
            "model": "legacy-global-model",
            "model_roles": {
                "review": {"provider": "unavailable", "model": "review-model-id"}
            },
        }
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=cfg),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = dt.delegate_task(
                goal="review", model_role="review", parent_agent=_make_mock_parent()
            )

        parsed = json.loads(out)
        assert "error" in parsed
        assert "review" in parsed["error"]
        assert MockAgent.call_count == 0

    def test_requesting_a_role_with_no_roles_configured_fails(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value={"max_iterations": 40}),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = dt.delegate_task(
                goal="review", model_role="review", parent_agent=_make_mock_parent()
            )

        parsed = json.loads(out)
        assert "error" in parsed
        assert "model_roles" in parsed["error"]
        assert MockAgent.call_count == 0


# ---------------------------------------------------------------------------
# 4. 結果紀錄：角色 + 解析後 provider/model + 實際執行身分
# ---------------------------------------------------------------------------

class TestModelRoleAppearsInResults:
    def test_results_carry_role_and_resolved_identity(self):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory),
        ):
            out = dt.delegate_task(
                tasks=[
                    {"goal": "plain"},
                    {"goal": "reviewed", "model_role": "review"},
                ],
                parent_agent=_make_mock_parent(),
            )

        results = {r["task_index"]: r for r in json.loads(out)["results"]}
        assert results[0].get("model_role") is None
        reviewed = results[1]
        assert reviewed["model_role"] == "review"
        # 解析後（設定推導出來的）身分
        assert reviewed["model_role_resolved"]["provider"] == "openai"
        assert reviewed["model_role_resolved"]["model"] == "review-model-id"
        assert reviewed["model_role_resolved"]["reasoning_effort"] == "high"
        # 實際執行身分 —— 從 child agent 物件讀出，不是模型自述
        assert reviewed["model"] == "review-model-id"
        assert reviewed["provider"] == "openai"


# ---------------------------------------------------------------------------
# 5. 工具 schema 只暴露設定裡存在的角色
# ---------------------------------------------------------------------------

class TestModelRoleSchema:
    def test_enum_lists_only_configured_roles(self):
        import tools.delegate_tool as dt

        with patch.object(dt, "_load_config", return_value=_ROLES_CFG):
            overrides = dt._build_dynamic_schema_overrides()

        props = overrides["parameters"]["properties"]
        assert props["model_role"]["enum"] == ["review", "second_opinion"]
        task_props = props["tasks"]["items"]["properties"]
        assert task_props["model_role"]["enum"] == ["review", "second_opinion"]

    def test_absent_when_no_roles_configured(self):
        """沒配置角色時完全不佔 schema 版面（零 footprint）。"""
        import tools.delegate_tool as dt

        with patch.object(dt, "_load_config", return_value={"max_iterations": 40}):
            overrides = dt._build_dynamic_schema_overrides()

        props = overrides["parameters"]["properties"]
        assert "model_role" not in props
        assert "model_role" not in props["tasks"]["items"]["properties"]

    def test_static_schema_is_not_mutated_by_overrides(self):
        import tools.delegate_tool as dt

        with patch.object(dt, "_load_config", return_value=_ROLES_CFG):
            dt._build_dynamic_schema_overrides()

        static_props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model_role" not in static_props
        assert "model_role" not in static_props["tasks"]["items"]["properties"]


# ---------------------------------------------------------------------------
# 6. 走正式工具入口（registry.dispatch）的整合驗證
# ---------------------------------------------------------------------------

class TestModelRoleThroughToolEntrypoint:
    def test_registry_dispatch_forwards_model_role(self):
        """從 registry.dispatch（模型真正呼叫的入口）進去，角色要一路傳到 AIAgent。

        父代理 depth=1（orchestrator 的 worker 派工）→ 同步執行，方便直接斷言。
        """
        import tools.delegate_tool as dt
        from tools.registry import registry

        parent = _make_mock_parent(depth=1)
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch.object(dt, "_get_max_spawn_depth", return_value=3),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = registry.dispatch(
                "delegate_task",
                {
                    "tasks": [
                        {"goal": "verify contract", "model_role": "review"},
                    ]
                },
                parent_agent=parent,
            )

        assert MockAgent.call_count == 1, out
        assert MockAgent.call_args.kwargs["model"] == "review-model-id"
        assert MockAgent.call_args.kwargs["provider"] == "openai"

    def test_model_cannot_inject_endpoint_or_credentials(self):
        """模型只能給角色名；endpoint／api_key／model 這類欄位必須被忽略。"""
        import tools.delegate_tool as dt
        from tools.registry import registry

        parent = _make_mock_parent(depth=1)
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch.object(dt, "_get_max_spawn_depth", return_value=3),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            registry.dispatch(
                "delegate_task",
                {
                    "tasks": [
                        {
                            "goal": "sneaky",
                            "model_role": "review",
                            "base_url": "http://evil.invalid/v1",
                            "api_key": "stolen",
                            "model": "gpt-does-not-exist",
                            "provider": "evil",
                        }
                    ]
                },
                parent_agent=parent,
            )

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert kwargs["api_key"] == "key-for-openai"
        assert kwargs["model"] == "review-model-id"
        assert kwargs["provider"] == "openai"


# ---------------------------------------------------------------------------
# 7. 非同步（背景）完成事件仍帶回模型身分
# ---------------------------------------------------------------------------

@pytest.fixture
def _clean_completion_queue():
    def _drain():
        while not process_registry.completion_queue.empty():
            try:
                process_registry.completion_queue.get_nowait()
            except Exception:
                break

    _drain()
    yield
    _drain()


def _drain_one(timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


class TestAsyncCompletionCarriesModelRoles:
    def test_background_batch_completion_reports_roles(self, _clean_completion_queue):
        import tools.delegate_tool as dt

        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory),
        ):
            out = dt.delegate_task(
                tasks=[
                    {"goal": "plain"},
                    {"goal": "reviewed", "model_role": "review"},
                ],
                background=True,
                parent_agent=_make_mock_parent(),
            )
            assert json.loads(out)["status"] == "dispatched", out
            evt = _drain_one()

        assert evt is not None
        assert evt["type"] == "async_delegation"
        # 批次層級：每個任務選了哪個角色
        assert evt["model_roles"] == [None, "review"]
        # 每個任務的結果仍帶著解析後與實際的模型身分
        by_idx = {r["task_index"]: r for r in evt["results"]}
        assert by_idx[1]["model_role"] == "review"
        assert by_idx[1]["model"] == "review-model-id"

    def test_completion_block_renders_role_identity(self):
        from tools.process_registry import _format_async_delegation

        text = _format_async_delegation(
            {
                "type": "async_delegation",
                "delegation_id": "d1",
                "is_batch": True,
                "goals": ["plain", "reviewed"],
                "model_roles": [None, "review"],
                "status": "completed",
                "results": [
                    {"task_index": 0, "status": "completed", "summary": "a"},
                    {
                        "task_index": 1,
                        "status": "completed",
                        "summary": "b",
                        "model_role": "review",
                        "model": "review-model-id",
                        "provider": "openai",
                        "model_role_resolved": {
                            "provider": "openai",
                            "model": "review-model-id",
                            "reasoning_effort": "high",
                        },
                    },
                ],
            }
        )
        assert "review" in text
        assert "review-model-id" in text


# ---------------------------------------------------------------------------
# 8. 真實解析鏈 E2E：真的 config.yaml → 真的 _load_config / 憑證解析 → child
# ---------------------------------------------------------------------------

class TestModelRoleEndToEndAgainstRealConfig:
    """只 mock 掉 AIAgent（避免真實模型連線與費用），其餘全部走真品。

    角色刻意用「直連 OpenAI 相容 endpoint」的形式（base_url + api_key），
    因為那條解析路徑不需要任何供應商憑證查詢，也不會發出網路請求。
    """

    def _write_config(self, tmp_path, monkeypatch):
        import hermes_cli.config as hcfg

        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "delegation:\n"
            "  max_iterations: 7\n"
            "  model_roles:\n"
            "    review:\n"
            "      base_url: \"http://127.0.0.1:19999/v1\"\n"
            "      api_key: \"role-key\"\n"
            "      model: \"configured-review-model\"\n"
            "      reasoning_effort: high\n"
            "    leaf:\n"          # 保留字：必須被拒絕，不能出現在 enum
            "      provider: openai\n"
            "      model: \"should-be-ignored\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        # 清掉 config 快取，確保讀到剛寫的檔案（快取鍵含 mtime，但路徑變了也要乾淨）
        hcfg._LOAD_CONFIG_CACHE.clear()
        hcfg._RAW_CONFIG_CACHE.clear()
        return home

    def test_real_config_feeds_schema_and_child_construction(
        self, tmp_path, monkeypatch
    ):
        import tools.delegate_tool as dt
        from tools.registry import registry

        self._write_config(tmp_path, monkeypatch)

        # 真的 _load_config() 讀到角色，且保留字被擋掉
        cfg = dt._load_config()
        assert cfg.get("max_iterations") == 7
        assert dt.available_model_role_names(cfg) == ["review"]

        # 工具 schema（模型真正看到的那份）只列出可用角色
        defs = registry.get_definitions({"delegate_task"}, quiet=True)
        schema = next(
            d["function"] for d in defs
            if d["function"]["name"] == "delegate_task"
        )
        props = schema["parameters"]["properties"]
        assert props["model_role"]["enum"] == ["review"]
        assert props["tasks"]["items"]["properties"]["model_role"]["enum"] == ["review"]

        # 走正式工具入口，真實憑證解析鏈 → 實際 child 建立路徑
        parent = _make_mock_parent(depth=1)
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_get_max_spawn_depth", return_value=3),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = registry.dispatch(
                "delegate_task",
                {"tasks": [{"goal": "review it", "model_role": "review"}]},
                parent_agent=parent,
            )

        assert MockAgent.call_count == 1, out
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["model"] == "configured-review-model"
        assert kwargs["base_url"] == "http://127.0.0.1:19999/v1"
        assert kwargs["api_key"] == "role-key"
        assert kwargs["provider"] == "custom"
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
        # 主代理完全沒被動到
        assert parent.model == "parent-model"
        assert parent.provider == "openai"

        result = json.loads(out)["results"][0]
        assert result["model_role"] == "review"
        assert result["model_role_resolved"]["model"] == "configured-review-model"
        assert result["model_role_resolved"]["reasoning_effort"] == "high"

    def test_real_config_unknown_role_fails_loudly(self, tmp_path, monkeypatch):
        import tools.delegate_tool as dt
        from tools.registry import registry

        self._write_config(tmp_path, monkeypatch)
        parent = _make_mock_parent(depth=1)
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_get_max_spawn_depth", return_value=3),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            out = registry.dispatch(
                "delegate_task",
                {"tasks": [{"goal": "x", "model_role": "leaf"}]},
                parent_agent=parent,
            )

        parsed = json.loads(out)
        assert "error" in parsed
        assert "reserved" in parsed["error"]
        assert MockAgent.call_count == 0

    def test_real_config_without_roles_keeps_schema_clean(self, tmp_path, monkeypatch):
        """沒有 model_roles 的使用者，schema 不該多出任何欄位。"""
        import hermes_cli.config as hcfg
        from tools.registry import registry

        home = tmp_path / "plain-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "delegation:\n  max_iterations: 11\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        hcfg._LOAD_CONFIG_CACHE.clear()
        hcfg._RAW_CONFIG_CACHE.clear()

        defs = registry.get_definitions({"delegate_task"}, quiet=True)
        schema = next(
            d["function"] for d in defs
            if d["function"]["name"] == "delegate_task"
        )
        props = schema["parameters"]["properties"]
        assert "model_role" not in props
        assert "model_role" not in props["tasks"]["items"]["properties"]


# ---------------------------------------------------------------------------
# 9. 指名模型不得被 fallback chain 偷換
# ---------------------------------------------------------------------------

class TestModelRoleDoesNotSilentlyFallBack:
    def test_role_pinned_child_gets_no_fallback_chain(self):
        """角色指名的模型不可用時必須失敗，不能改用父代理 fallback 鏈上的模型。"""
        import tools.delegate_tool as dt

        parent = _make_mock_parent()
        parent._fallback_chain = [
            {"provider": "openrouter", "model": "some-other-model"}
        ]
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(
                goal="review", model_role="review", parent_agent=parent
            )

        assert MockAgent.call_args.kwargs["fallback_model"] is None

    def test_task_without_role_still_inherits_fallback_chain(self):
        """沒選角色的任務，fallback 繼承行為保持不變（回歸防護）。"""
        import tools.delegate_tool as dt

        chain = [{"provider": "openrouter", "model": "some-other-model"}]
        parent = _make_mock_parent()
        parent._fallback_chain = chain
        factory, _created = _mock_agent_factory()
        with (
            patch.object(dt, "_load_config", return_value=_ROLES_CFG),
            patch.object(dt, "_resolve_delegation_credentials", _fake_resolver),
            patch("run_agent.AIAgent", side_effect=factory) as MockAgent,
        ):
            dt.delegate_task(goal="plain work", parent_agent=parent)

        assert MockAgent.call_args.kwargs["fallback_model"] == chain
