# PR12 Cognition Observation Benchmark + Persisted Trace Verification 實作計畫

## Base branch

- Worktree: `~/.hermes/hermes-agent/.worktrees/pr12-cognition-observation-benchmark`
- Branch: `feat/cognition-observation-benchmark-pr12`
- Base: `feat/status-route-fix-pr11` (`c19d49c2`)

## 目標

建立一組固定、可重跑、純 deterministic 的 cognition observation benchmark，用來驗證 PR1–PR11 的 route / interaction / uncertainty / verification / trace / trajectory export 行為沒有漂移。

PR12 不是新認知策略，而是「觀測與回歸測試基準」。它讓後續 PR13+（memory ranker、metadata、policy memory）不會盲飛。

## 非目標

- 不修改 prompt construction。
- 不修改 memory schema。
- 不修改 model dispatch / tool dispatch。
- 不做 online learning / calibration。
- 不跑真實外部 API。
- 不要求模型內容品質；只驗 cognition metadata 和 persisted trace artifacts。

## 設計

新增純 helper module：

- `agent/cognition_observation_benchmark.py`

核心資料結構：

```python
@dataclass(frozen=True)
class CognitionObservationCase:
    name: str
    prompt: str
    expected_mode: str
    expected_dialogue_mode: str
    expected_answer_density: str
    expected_retrieval_plan: str | None = None
    expected_verification_plan: str | None = None
    expected_allow_cheap_model: bool | None = None
    expected_uncertainty_action: str | None = None
```

固定 case set：

1. `fast_query`
   - prompt: `what time is it in tokyo?`
   - expected: `fast`, `query`, `brief`, cheap allowed
2. `project_status`
   - prompt: `目前 cognition stack 狀態如何？`
   - expected: `standard`, `status`, `brief`, cheap blocked
3. `history_lookup`
   - prompt: `上次我們討論的 root cause 是什麼？`
   - expected: `deep`, route historical, verification full
4. `debate`
   - prompt: `你覺得這個架構取捨應不應該反駁？`
   - expected: `deep` 或 architecture/debate path, `debate`, `standard`
5. `execution`
   - prompt: `幫我做 PR12，跑測試後 commit`
   - expected: `deep`, `execution`, `brief`

Helper functions：

```python
def cognition_observation_cases() -> tuple[CognitionObservationCase, ...]
def expected_case_names() -> tuple[str, ...]
def evaluate_cognition_trace(case, trace) -> dict[str, Any]
def evaluate_cognition_traces(traces_by_case) -> dict[str, Any]
```

`evaluate_cognition_trace` 不 raise，回傳 JSON-friendly result：

```json
{
  "name": "project_status",
  "passed": true,
  "failures": [],
  "observed": {...}
}
```

這讓 CLI/測試/未來 telemetry 都能共用。

## Persisted trace verification

PR8 已把 `metadata.cognition_trace` 寫入 trajectory JSONL。PR12 要加 runtime test，確定固定 prompt 真的會：

- 產生 result[`cognition_trace`]
- `_current_turn_cognition_metadata["cognition_trace"]` 存在
- save trajectory 時 metadata 包含 `{"cognition_trace": trace}`
- offline report 可以從 JSONL 讀出 trace counters

測試可用 patch/mock，不打真實 API：

- 用既有 `_run_one_turn` fake response 驅動 runtime。
- 對 trajectory export 可直接呼叫 `_save_trajectory` 並 patch `run_agent._save_trajectory_to_file`，或用 temp JSONL 呼叫 `agent.trajectory.save_trajectory` 後 `analyze_cognition_trace_jsonl`。

## TDD / 測試計畫

先加 failing tests：

1. `tests/agent/test_cognition_observation_benchmark.py`
   - case names fixed and ordered
   - evaluate passes for matching trace
   - evaluate reports failures for mismatched trace
   - evaluate handles missing/malformed trace without raising

2. `tests/run_agent/test_cognition_observation_benchmark.py` 或放入 `tests/run_agent/test_run_agent.py`
   - fixed cases produce expected runtime cognition_trace
   - persisted trajectory JSONL contains `metadata.cognition_trace`
   - offline report sees trace present and route/dialogue counters

建議新增獨立檔案，避免 `test_run_agent.py` 繼續膨脹。

Targeted command：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_cognition_observation_benchmark.py \
  tests/run_agent/test_cognition_observation_benchmark.py \
  tests/agent/test_cognitive_router.py \
  tests/agent/test_cognition_trace_report.py \
  -q
```

Smoke：

```bash
~/.hermes/hermes-agent/venv/bin/python - <<'PY'
import agent.cognition_observation_benchmark
import run_agent
print('import smoke ok')
PY
```

## Commit split

1. `docs: plan cognition observation benchmark PR12`
2. `test: cover cognition observation benchmark`
3. `feat: add cognition observation benchmark helpers`
4. `test: verify persisted cognition trace benchmark artifacts` 或合併到第 2/3 視 diff 大小

## 安全性 / 相容性

- 新 module 純函式、無 IO、無外部 API。
- runtime test 使用 mock/fake，不改 production loop。
- trajectory verification 只驗 PR8 已存在 export path。
- 不引入新的必跑耗時 benchmark；只是 pytest regression fixtures。
