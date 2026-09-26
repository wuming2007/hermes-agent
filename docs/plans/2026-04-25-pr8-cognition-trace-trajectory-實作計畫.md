# PR8：Cognition Trace Trajectory Export 實作計畫

> **For Hermes:** Use hermes-cognitive-stack-pr-development and test-driven-development. This PR is a small chained worktree PR based on PR7.

**Goal:** 把 PR7 產生的 `cognition_trace` 安全寫入 trajectory JSONL entry，讓後續離線分析 / calibration / telemetry 可以讀到每回合的認知路由、uncertainty、verification trace。

**Architecture:** PR8 只做 trajectory export plumbing。`agent.trajectory.save_trajectory(...)` 新增可選 metadata 參數，透過純函式把 metadata 正規化成 JSON-friendly entry metadata；`run_agent.AIAgent._save_trajectory(...)` 在 `save_trajectories=True` 時把當前 turn 的 `cognition_trace` 傳入。所有舊呼叫保持相容，trajectory 寫檔失敗仍然 fail-open。

**Base branch:** `feat/cognition-turn-trace-pr7`

**Branch:** `feat/cognition-trace-trajectory-pr8`

**Worktree:** `/Users/wuming/.hermes/hermes-agent/.worktrees/pr8-cognition-trace-trajectory`

---

## 非目標

- 不做 learned calibration / online adaptation。
- 不改 memory schema。
- 不改 prompt / system prompt / model dispatch。
- 不新增背景 telemetry uploader。
- 不改 trajectory conversation format；只在 JSONL entry top-level 增加 metadata。

## 相容性與安全要求

- `save_trajectory(trajectory, model, completed)` 舊呼叫必須仍然有效。
- 未提供 metadata 時，輸出的 JSONL entry 不能多出 `metadata` key。
- metadata 只接受 JSON-friendly copy，不應保留 caller 物件 reference。
- 若 metadata 含有 non-serializable value，應轉成 string 或安全降級，不能讓 trajectory save 因 metadata 爆掉。
- `run_agent` 中 trace export failure 不得影響 final response。

## 新增資料形狀

Trajectory JSONL entry 原本：

```json
{
  "conversations": [...],
  "timestamp": "...",
  "model": "...",
  "completed": true
}
```

PR8 在有 cognition trace 時新增：

```json
{
  "conversations": [...],
  "timestamp": "...",
  "model": "...",
  "completed": true,
  "metadata": {
    "cognition_trace": {
      "schema_version": 1,
      "enabled": true,
      "route": {...},
      "uncertainty": {...},
      "verification": {...}
    }
  }
}
```

## Task 1：Trajectory metadata pure helper + unit tests

**Objective:** 在 `agent/trajectory.py` 中新增純函式，把可選 metadata 正規化成可 JSON dump 的 metadata dict，並讓 `save_trajectory` 支援 `metadata=None`。

**Files:**
- Modify: `agent/trajectory.py`
- Create: `tests/agent/test_trajectory_metadata.py`

**TDD steps:**

1. Write failing tests first:
   - no metadata → JSONL entry 沒有 `metadata` key。
   - cognition trace metadata → entry 有 `metadata.cognition_trace`。
   - input metadata 不被 mutate。
   - non-serializable metadata value 會安全轉成 string。
2. Run RED:

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/agent/test_trajectory_metadata.py -q
```

Expected: fail because new helper / metadata argument does not exist yet.

3. Implement minimal:
   - add helper such as `build_trajectory_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None`
   - update `save_trajectory(..., metadata: Mapping[str, Any] | None = None)`
   - include `entry["metadata"] = normalized_metadata` only when non-empty.
4. Run GREEN same command.
5. Commit:

```bash
git add agent/trajectory.py tests/agent/test_trajectory_metadata.py
git commit -m "feat: add trajectory metadata export helper"
```

## Task 2：Runtime wiring for cognition trace trajectory export

**Objective:** 讓 `AIAgent._save_trajectory` 在 trajectory enabled 時，把 PR7 的 current turn cognition trace 寫進 trajectory metadata。

**Files:**
- Modify: `run_agent.py`
- Modify: `tests/run_agent/test_run_agent.py`

**TDD steps:**

1. Add failing runtime test:
   - enable `agent.save_trajectories = True`
   - set/produce `_current_turn_cognition_metadata["cognition_trace"]`
   - patch `_save_trajectory_to_file`
   - call `agent._save_trajectory(...)`
   - assert third-party save receives `metadata={"cognition_trace": trace}`
2. Add compatibility test:
   - no cognition trace → still calls `_save_trajectory_to_file(trajectory, model, completed)` behavior-compatible; with new signature metadata may be omitted or `None`.
3. Run RED targeted test selectors.
4. Implement minimal change in `_save_trajectory`:

```python
metadata = None
if isinstance(self._current_turn_cognition_metadata, dict):
    trace = self._current_turn_cognition_metadata.get("cognition_trace")
    if trace is not None:
        metadata = {"cognition_trace": trace}
_save_trajectory_to_file(trajectory, self.model, completed, metadata=metadata)
```

5. Run GREEN targeted runtime tests.
6. Commit:

```bash
git add run_agent.py tests/run_agent/test_run_agent.py
git commit -m "feat: export cognition trace in trajectories"
```

## Final verification

Run targeted PR8 confidence suite:

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_trajectory_metadata.py \
  tests/agent/test_cognition_trace.py \
  tests/agent/test_consistency_guard.py \
  tests/agent/test_uncertainty_policy.py \
  tests/run_agent/test_run_agent.py::TestConsistencyGuardWiring \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

Then verify:

```bash
git status --short --branch
git log --oneline --decorate -6
```

## Commit split

1. `docs: plan cognition trace trajectory PR8`
2. `feat: add trajectory metadata export helper`
3. `feat: export cognition trace in trajectories`
