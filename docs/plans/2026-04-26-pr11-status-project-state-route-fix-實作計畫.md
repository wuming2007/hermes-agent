# PR11 Status / Project-State Route Fix 實作計畫

## Base branch

- Worktree: `~/.hermes/hermes-agent/.worktrees/pr11-status-route-fix`
- Branch: `feat/status-route-fix-pr11`
- Base: `feat/interaction-stance-pr10` (`82d1eb5d`)

## 問題

PR10 已能把「目前 cognition stack 狀態如何？」標成：

- `dialogue_mode = status`
- `answer_density = brief`

但 cognitive route 本身仍可能因短句落入：

- `mode = fast`
- `retrieval_plan = principles_only`
- `verification_plan = none`
- `allow_cheap_model = True`
- `routing_reasons = ["short_simple"]`

這對 project-state / work-status 類問題是錯的：狀態問題通常需要查 repo、daily note、recent sessions、trajectory/report 或至少不要便宜模型隨口答。

## 目標

新增 deterministic project-status route guard：

- 短的 project-state/status 問題不得被 `short_simple` 吞掉。
- route 至少提升到 `standard`。
- retrieval 至少 `principles_plus_semantic`。
- verification 至少 `light`。
- `allow_cheap_model = False`。
- 保留 PR10 stance metadata：`dialogue_mode=status`、`answer_density=brief`。
- 加上 routing reason，例如 `status_lookup:<keyword>` 或 `project_state:<keyword>`。

## 非目標

- 不修改 prompt construction。
- 不修改 memory schema。
- 不修改 tool dispatch / model dispatch。
- 不在此 PR 實作真正的 daily-note/session-search 強制工具呼叫；這只修 cognitive route，讓 runtime 不走 cheap/fast 並啟用 standard retrieval/verification metadata。
- 不改一般 fast query，例如 `ping`、`what time is it in tokyo?`。

## 設計

在 `agent/cognitive_router.py` 加一個 pure guard，順序放在：

1. deep triggers 後
2. empty message 前 / fast eligibility 前

因為狀態問題不是深度架構推理，也不應該被 fast short-simple 接走。

建議 helper：

```python
_STATUS_ROUTE_KEYWORDS = (...)
_PROJECT_STATE_KEYWORDS = (...)

def _status_lookup_reason(text_lower: str) -> str | None:
    ...
```

觸發範圍：

- status/progress/state/currently/where are we/what remains
- 目前/狀態/進度/做到哪/剩下/還差/完成了嗎
- 搭配 project/repo/PR/stack/worktree/branch/cognition/Hermes/任務/專案/分支/工作樹/實作 等 project-state cues
- 中文短句如「PR11 狀態？」、「目前 cognition stack 狀態如何？」應觸發。

輸出：

```python
CognitiveRoute(
    mode="standard",
    retrieval_plan="principles_plus_semantic",
    verification_plan="light",
    allow_cheap_model=False,
    consistency_check=False,
    routing_reasons=[reason],
)
```

然後照 PR10 現有 `_apply_interaction_stance(...)` 附加 stance。

## TDD / 測試計畫

先加 failing tests，確認現在會失敗：

- `tests/agent/test_cognitive_router.py`
  - `test_project_status_prompt_routes_standard_not_fast`
  - `test_pr_status_prompt_routes_standard_not_fast`
  - `test_regular_short_query_stays_fast`

- `tests/run_agent/test_run_agent.py::TestCognitiveRouting`
  - `test_status_project_prompt_routes_standard_and_blocks_cheap_model`

預期 PR11 後：

- status/project prompt:
  - `mode == "standard"`
  - `retrieval_plan == "principles_plus_semantic"`
  - `verification_plan == "light"`
  - `allow_cheap_model is False`
  - `dialogue_mode == "status"`
  - `answer_density == "brief"`
  - routing reason 包含 `status_lookup` 或 `project_state`

Targeted tests：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_cognitive_router.py \
  tests/agent/test_interaction_stance.py \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

Confidence / smoke：

```bash
~/.hermes/hermes-agent/venv/bin/python - <<'PY'
import agent.cognitive_router
import agent.interaction_stance
import run_agent
print('import smoke ok')
PY
```

## Commit split

1. `docs: plan status route fix PR11`
2. `test: cover project status route fix`
3. `feat: route project status prompts to standard lookup`

## 安全性 / 相容性

- cognition disabled 仍 return `None`。
- 一般 short-simple prompt 仍 fast。
- deep triggers 仍優先於 status guard。
- 不新增 tool/model side effects。
- metadata-only stance 保持 PR10 行為。
