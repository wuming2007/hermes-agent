# PR12.1 CLI save_trajectories config wiring 實作計畫

## Base

- Base branch: `runtime/active-cognitive-stack @ dbbe27e7`
- Worktree: `~/.hermes/hermes-agent/.worktrees/pr12-1-cli-trajectory-config`
- Branch: `feat/cli-trajectory-config-pr12-1`

## 問題

`~/.hermes/config.yaml` 已設 `save_trajectories: true`，但一般 interactive CLI 建立 `AIAgent(...)` 時沒有把此設定傳入 `save_trajectories` 參數，導致 `AIAgent.__init__` 使用預設 `False`，不會產生 `trajectory_samples.jsonl`，也就沒有 persisted `metadata.cognition_trace` 作為 PR12 後續觀測資料。

## 目標

1. 讓 CLI runtime 讀取 config 中的 trajectory 開關。
2. 支援既有 root-level `save_trajectories: true`。
3. 支援較結構化的 `agent.save_trajectories: true`，且優先於 root-level。
4. 將解析後的 boolean 傳給 `AIAgent(save_trajectories=...)`。
5. 不改 prompt construction、memory schema、model dispatch、tool dispatch。

## 非目標

- 不改 trajectory JSONL schema。
- 不改 cognition trace schema。
- 不啟用任何線上學習或自動 memory promotion。
- 不改 gateway 行為；本 PR 只修 CLI interactive path。

## 預計修改檔案

- `cli.py`
  - 在 `HermesCLI.__init__` 中解析 `self.save_trajectories`。
  - 在 `_init_agent()` 建立 `AIAgent(...)` 時傳入 `save_trajectories=self.save_trajectories`。
- `tests/cli/test_cli_trajectory_config.py` 或既有 CLI 測試檔
  - test root-level config true 會傳到 AIAgent。
  - test `agent.save_trajectories` override root-level。
  - test 缺省保持 False。

## TDD 驗證

RED：新增測試後先跑，應該因為 `AIAgent` 呼叫 kwargs 沒有 `save_trajectories=True` 而失敗。

GREEN：補 CLI wiring 後測試通過。

Targeted tests：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/cli/test_cli_trajectory_config.py -q
```

Broader tests：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/cli/test_cli_trajectory_config.py tests/agent/test_cognition_observation_benchmark.py tests/run_agent/test_cognition_observation_benchmark.py -q
```

Runtime smoke after integration：

- `cognition.enabled=True`
- `save_trajectories root=True`
- `HermesCLI().save_trajectories=True`
- `HermesCLI._init_agent()` passes `save_trajectories=True` to `AIAgent`
- import smoke passes

## Commit split

1. `docs: plan cli trajectory config wiring PR12.1`
2. `test: cover CLI save trajectories config wiring`
3. `fix: wire CLI save trajectories config into AIAgent`
