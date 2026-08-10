# PR17 Plasticity Promotion / Decay v1 實作計畫

## Base

- Base branch: `runtime/active-cognitive-stack`
- Base commit: `b2b8cfe0 feat: add deterministic process monitor`
- Feature branch: `feat/plasticity-promotion-decay-pr17`
- Worktree: `~/.hermes/hermes-agent/.worktrees/pr17-plasticity`

## 目標

PR17 要把 PR13/PR14 的 memory ranking metadata 往「可塑性」推進：讓記憶候選項可根據成功、糾正、驗證、年齡等訊號，產生 deterministic 的 promotion / decay / supersede 建議。

這一版只做 deterministic metadata layer，不直接改寫記憶檔、不做線上學習、不刪資料。

## 非目標

- 不做 LLM-based reflection / learning。
- 不直接修改 MEMORY.md / USER.md / provider storage。
- 不做自動刪除。
- 不改變現有 memory provider schema 的必填欄位。
- 不讓 plasticity failure 影響正常 ranked prefetch。

## 新增模組

`agent/memory_plasticity.py`

建議 dataclasses：

- `PlasticitySignal`
  - `success_count`
  - `correction_count`
  - `verification_count`
  - `days_since_verified`
  - `explicit_decay`
  - `superseded_by`
  - `confidence_delta`
- `PlasticityDecision`
  - `action`: `promote | maintain | decay | supersede`
  - `promotion_delta`
  - `decay_delta`
  - `reinforcement_delta`
  - `confidence_delta`
  - `status`
  - `reasons`
- `PlasticityConfig`
  - bounded threshold / delta config。

Pure functions：

- `normalize_plasticity_signal(value) -> PlasticitySignal`
- `resolve_plasticity_decision(signal, config=None) -> PlasticityDecision`
- `apply_plasticity_to_candidate(candidate, signal, config=None) -> MemoryCandidate`
- `build_plasticity_metadata(decisions) -> dict`
- `build_plasticity_context(decisions) -> str`

## 行為規則

1. success / verification 增加 reinforcement、confidence，降低 decay。
2. correction / explicit_decay / stale verified age 增加 decay。
3. superseded_by 直接標記 `supersede`，metadata status 變 `superseded`。
4. 所有 numeric signal clamp 到合理範圍。
5. apply 時保留原始 candidate 欄位，只用 `dataclasses.replace` 回傳新 candidate。
6. candidate metadata 必須 normalize 成 `MemoryObjectMetadata`，並保留 source_trace / notes。
7. 如果 metadata 已有 notes，新增 `plasticity_action` / `plasticity_reasons`。
8. 不做 destructive deletion。

## Runtime 接線

先只接在 `MemoryManager.prefetch_ranked_for_policy(...)` 的 candidate normalization/ranking 前：

- provider 若沒有 plasticity metadata，維持舊行為。
- candidate.metadata.notes 可帶 `plasticity` mapping 作為 signal。
- MemoryManager 對每個 candidate 嘗試 apply；失敗就保留原 candidate。
- 增加 `last_plasticity_metadata`，供 runtime/trace 可讀。
- 若無 candidates 或無 plasticity signal，metadata 應是 disabled/empty。

## Trace metadata

Flat keys：

- `plasticity_enabled`
- `plasticity_decision_count`
- `plasticity_actions`
- `plasticity_promoted_count`
- `plasticity_decayed_count`
- `plasticity_superseded_count`

Nested trace block：

```python
"plasticity": {
    "enabled": bool,
    "decision_count": int,
    "actions": list[str],
    "promoted_count": int,
    "decayed_count": int,
    "superseded_count": int,
}
```

## 測試

先 RED：

- `tests/agent/test_memory_plasticity.py`
  - normalize signal。
  - promote / maintain / decay / supersede decision。
  - apply candidate with bounded signals。
  - metadata/context output。
- `tests/agent/test_memory_manager.py`
  - candidate with plasticity notes gets promoted/decayed before ranking。
  - failures are fail-open。
  - `last_plasticity_metadata` updated。
- `tests/agent/test_cognition_trace.py`
  - default plasticity block。
  - grouped plasticity metadata。

Targeted command：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_memory_plasticity.py \
  tests/agent/test_memory_ranker.py \
  tests/agent/test_memory_manager.py \
  tests/agent/test_memory_provider.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

## Commit split

1. `docs: plan memory plasticity PR17`
2. `test: cover memory plasticity PR17`
3. `feat: add deterministic memory plasticity layer`

## 安全性

- deterministic only。
- fail-open。
- additive metadata。
- 不刪記憶、不改 provider storage。
