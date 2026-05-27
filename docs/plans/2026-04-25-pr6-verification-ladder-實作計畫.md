# PR6：Verification Ladder 實作計畫

> Base：`feat/uncertainty-policy-pr5`。本 PR 不重寫 PR1~PR5，只把 PR3 的 `light/full` consistency guard 升級成可觀測、可逐步擴充的 verification ladder。

## 目標

把目前的單層 guard dispatch：

- `none`：不驗
- `light`：本地 rule-based check
- `full`：一次 slow verifier call

整理成明確 ladder：

1. `self_correction`：本地、無外部 call，做極低成本自檢/空回覆/明顯矛盾檢查。
2. `fast_monitor`：本地 rule-based monitor，等價承接既有 `light` guard 行為。
3. `slow_verifier`：外部 verifier call，等價承接既有 `full` guard 行為。

第一版 PR6 先做 deterministic planner + runtime metadata，不引入多次 LLM repair loop。

## 非目標

- 不新增第二個 LLM verifier call。
- 不改 system prompt / prompt cache prefix。
- 不把 verifier internal reasoning 寫入 messages。
- 不做 learned calibration / telemetry adaptation。
- 不改 memory schema。

## 設計原則

- Disabled cognition 保持 no-op。
- Guard/laddder 任何 exception 都 fail-open，保留 candidate response。
- 舊 `verification_plan` contract 保持向後相容：`none/light/full` 仍是路由輸入。
- 新 metadata 要下游友善，可被 trajectory / future telemetry 使用。

## Task 1：純函式 ladder planner

Files：
- Modify：`agent/consistency_guard.py`
- Modify：`tests/agent/test_consistency_guard.py`

新增：
- `VerificationStage = Literal["self_correction", "fast_monitor", "slow_verifier"]`
- frozen dataclass `VerificationLadderPlan`
- `resolve_verification_ladder(cognition_route) -> VerificationLadderPlan`

預期 mapping：
- no route / `verification_plan="none"` / unknown plan：`enabled=False`, stages=()
- `light`：`enabled=True`, stages=("self_correction", "fast_monitor")
- `full`：`enabled=True`, stages=("self_correction", "fast_monitor", "slow_verifier")

Task 1 不改 runtime dispatch，只讓 planner 可單測。

## Task 2：runtime wiring + metadata

Files：
- Modify：`run_agent.py`
- Modify：`tests/run_agent/test_run_agent.py`

做法：
- `run_agent.py` guard dispatch 先呼叫 `resolve_verification_ladder(...)`。
- 仍維持舊行為：
  - `light` route 只跑既有 local light check。
  - `full` route 跑既有 full verifier path。
- 新增 metadata：
  - `verification_ladder_enabled`
  - `verification_ladder_stages`
  - `verification_ladder_executed_stages`
  - `verification_ladder_terminal_stage`

第一版 executed stages 可反映實際跑到的 guard：
- light：`["self_correction", "fast_monitor"]`
- full：`["self_correction", "fast_monitor", "slow_verifier"]`

## Task 3：回歸測試

Targeted tests：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_consistency_guard.py \
  tests/agent/test_uncertainty_policy.py \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  tests/run_agent/test_run_agent.py::TestConsistencyGuardWiring \
  -q
```

## Commit 切法

1. `docs: plan verification ladder PR6`
2. `feat: add verification ladder planner`
3. `feat: expose verification ladder metadata`
