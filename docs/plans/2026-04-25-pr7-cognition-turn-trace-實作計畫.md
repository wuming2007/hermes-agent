# PR7：Cognition Turn Trace 實作計畫

> **For Hermes:** 以 chained worktree 小 PR 方式實作；使用 TDD，任務完成後分段 commit。這個 PR 只做 deterministic trace snapshot，不做 learned adaptation。

**Goal:** 把 PR5 / PR6 產出的 cognitive route、uncertainty policy、verification ladder / guard metadata 正規化成穩定、可測、可下游消費的 turn trace snapshot。

**Architecture:** 新增純函式 trace builder，從既有 flat cognition metadata 建立 nested trace dict；runtime 只在安全點附加 `cognition_trace`，並保留舊 flat keys 以維持相容性。trace 建立失敗必須 fail-open，不影響主回合。

**Base:** `feat/verification-ladder-pr6`

**Branch:** `feat/cognition-turn-trace-pr7`

**Worktree:** `/Users/wuming/.hermes/hermes-agent/.worktrees/pr7-cognition-turn-trace`

---

## 背景

PR5 已加入 uncertainty / competence policy，PR6 已加入 deterministic verification ladder metadata。這些資訊目前散落在 `_current_turn_cognition_metadata` 的 flat keys：

- route：`mode`, `retrieval_plan`, `verification_plan`, `allow_cheap_model`, `consistency_check`, `routing_reasons`
- uncertainty：`original_mode`, `uncertainty_confidence_band`, `uncertainty_action`, `uncertainty_reasons`, `depth_escalated`, `target_mode`, `require_tool_evidence`, `seek_human`
- verification ladder：`verification_ladder_enabled`, `verification_ladder_source_plan`, `verification_ladder_stages`, `verification_ladder_applied_stages`
- verification result：`verification_applied`, `verification_changed`, `verification_notes`

PR7 要做的是把這些資料整理成穩定 trace schema，作為未來 telemetry / calibration / trajectory integration 的地基。

## 非目標

- 不做 learned calibration。
- 不做 telemetry-driven adaptation。
- 不改 memory provider schema。
- 不改 prompt / system prompt / prompt cache prefix。
- 不新增 verifier LLM call。
- 不移除既有 flat metadata keys。
- 不讓 trace builder 失敗中斷主 run loop。

## Trace schema v1

新增 nested key：

```python
metadata["cognition_trace"] = {
    "schema_version": 1,
    "enabled": True | False,
    "route": {
        "mode": str,
        "original_mode": str | None,
        "retrieval_plan": str | None,
        "verification_plan": str | None,
        "allow_cheap_model": bool | None,
        "consistency_check": bool | None,
        "routing_reasons": list[str],
    },
    "uncertainty": {
        "present": bool,
        "confidence_band": str | None,
        "action": str | None,
        "reasons": list[str],
        "depth_escalated": bool,
        "target_mode": str | None,
        "require_tool_evidence": bool,
        "seek_human": bool,
    },
    "verification": {
        "ladder_enabled": bool,
        "ladder_source_plan": str | None,
        "ladder_stages": list[str],
        "ladder_applied_stages": list[str],
        "applied": bool | None,
        "changed": bool | None,
        "notes": list[str],
    },
}
```

Disabled cognition trace:

```python
{
    "schema_version": 1,
    "enabled": False,
    "route": {"mode": "disabled", ...},
    "uncertainty": {"present": False, ...},
    "verification": {"ladder_enabled": False, ...},
}
```

## Task 1：新增 deterministic cognition trace builder

**Objective:** 新增純函式，從 flat metadata 正規化 trace dict，不接觸 runtime。

**Files:**
- Create: `agent/cognition_trace.py`
- Create: `tests/agent/test_cognition_trace.py`

**API:**

```python
def build_cognition_turn_trace(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    ...
```

**TDD cases:**

1. `None` / non-dict / disabled metadata 建立 disabled trace。
2. Route metadata 正規化到 `route`。
3. Uncertainty keys 存在時 `uncertainty.present == True`。
4. Uncertainty keys 不存在時 `present == False` 並有安全預設值。
5. Verification ladder / guard result keys 正規化到 `verification`。
6. 輸入 list/tuple/set reasons/stages 皆轉成 list[str]，非 list 值安全轉成空 list 或單值字串 list（測試至少涵蓋 tuple）。
7. Builder 回傳新 dict，不 mutate 原 metadata。

**RED command:**

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/agent/test_cognition_trace.py -q
```

Expected before implementation: import/module missing failure.

**GREEN command:**

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/agent/test_cognition_trace.py -q
```

Expected after implementation: all new tests pass.

**Commit:**

```bash
git add agent/cognition_trace.py tests/agent/test_cognition_trace.py
git commit -m "feat: add cognition turn trace builder"
```

## Task 2：接進 runtime metadata / result

**Objective:** 在 `run_agent.py` 的回合末端建立 trace snapshot，附加到 `_current_turn_cognition_metadata` 與回傳 result，且 fail-open。

**Files:**
- Modify: `run_agent.py`
- Modify: `tests/run_agent/test_run_agent.py`

**Runtime rule:**

- 在 verification guard / ladder metadata 完成之後、`result` dict 建立前，呼叫 `build_cognition_turn_trace(self._current_turn_cognition_metadata)`。
- 若 `_current_turn_cognition_metadata` 是 dict：
  - 寫入 `self._current_turn_cognition_metadata["cognition_trace"] = trace`
- `result` dict 加入：
  - `"cognition_trace": trace_or_none`
- 若 builder raise：
  - log warning
  - 不改 final_response
  - 不丟 exception
  - `result["cognition_trace"]` 可為 `None`

**TDD cases:**

1. Disabled cognition run returns result with disabled `cognition_trace`。
2. Standard / light route run returns trace with route + verification ladder metadata。
3. Uncertainty escalation route returns trace with `uncertainty.present == True` and `original_mode` preserved。
4. Builder exception does not break conversation; final response still returned and `cognition_trace` is `None` or absent-safe。

**Targeted command:**

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_cognition_trace.py \
  tests/agent/test_consistency_guard.py \
  tests/agent/test_uncertainty_policy.py \
  tests/run_agent/test_run_agent.py::TestConsistencyGuardWiring \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

**Commit:**

```bash
git add run_agent.py tests/run_agent/test_run_agent.py
git commit -m "feat: wire cognition turn trace metadata"
```

## 驗收標準

- PR7 branch clean。
- Trace builder unit tests pass。
- PR5 / PR6 targeted cognition tests pass。
- Runtime result exposes `cognition_trace` without breaking old flat metadata。
- All trace logic is deterministic and side-effect-free except runtime attachment。
- Fail-open behavior covered by test。
