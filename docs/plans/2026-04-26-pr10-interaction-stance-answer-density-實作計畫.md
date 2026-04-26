# PR10 Interaction Stance / Answer Density Routing 實作計畫

## Base branch

- Worktree: `~/.hermes/hermes-agent/.worktrees/pr10-interaction-stance`
- Branch: `feat/interaction-stance-pr10`
- Base: `feat/cognition-trace-report-pr9` (`e6489b56`)

## 目標

新增一層純 deterministic 的互動姿態路由，讓 cognition stack 除了 `fast / standard / deep`、retrieval、verification 之外，也能記錄「這一輪應該怎麼說話」。

新增 contract：

- `dialogue_mode`: `query | status | exploration | debate | execution`
- `answer_density`: `brief | standard | expanded`

預期效果：

- 查答案 / 狀態 → 精簡，不拖泥帶水。
- 探索想法 → 可以展開。
- 辯論 / 反駁 → 有主見，但不過度展開。
- 執行任務 → 少講、多做，回報 concrete result。

## 非目標

- 不改 prompt construction。
- 不改 memory schema。
- 不改 tool dispatch / model dispatch。
- 不做 learned calibration 或 online adaptation。
- 不在 PR10 處理 PR11 的「短 project-state 問題不要被 short_simple 吞掉」；PR10 只加 stance metadata。

## 設計

### 1. 純 planner

新增 `agent/interaction_stance.py`：

- dataclass `InteractionStance(dialogue_mode, answer_density, stance_reasons)` frozen。
- function `resolve_interaction_stance(user_message, cognition_route, routing_config, agent_state=None)`。
- cognition disabled 時仍可回傳 stance，因為這是 metadata/UX hint；但 runtime wiring 會只在 cognition route metadata 內附加，disabled path 保持安全。
- malformed config fail-safe：unknown override 值回預設，不 raise。

Heuristic 初版：

- `status`: 目前狀態、進度、status、what is the state、目前如何、做到哪、PR 狀態 → `brief`
- `execution`: 幫我做、修、改、commit、跑測試、整理、建立、寫入、delete/deploy/send/publish 等 → `brief`
- `debate`: 你覺得、反駁、不同意、評估、tradeoff、應不應該、pros/cons → `standard`
- `exploration`: brainstorm、聊聊、發想、設計哲學、探索、為什麼、怎麼看 → `expanded`
- fallback `query`: short/fast → `brief`; standard/deep → `standard`

### 2. router contract 擴充

在 `CognitiveRoute` 加 optional/default fields：

- `dialogue_mode: DialogueMode = "query"`
- `answer_density: AnswerDensity = "standard"`

保留舊 constructor 相容性；現有 tests 不應因新增欄位而壞。

`resolve_cognitive_route(...)` 在 return 前先決定基本 route，然後套 stance metadata。

### 3. runtime metadata wiring

在 `AIAgent._resolve_current_cognitive_route` flat metadata 加：

- `dialogue_mode`
- `answer_density`
- `stance_reasons`

uncertainty escalation 產生的新 `CognitiveRoute` 必須保留原 stance 欄位。

Fail-open：stance resolver exception 只 log warning，回到 query/standard 或保留原 route，不影響主 loop。

### 4. cognition trace

在 `agent/cognition_trace.py` 加 nested：

```json
"interaction": {
  "dialogue_mode": "...",
  "answer_density": "...",
  "stance_reasons": ["..."]
}
```

舊 flat route metadata 不移除。

### 5. offline report

PR9 analyzer 加 counters：

- `interaction.dialogue_modes`
- `interaction.answer_densities`
- `interaction.stance_reasons`

Malformed/missing trace 仍不可 raise。

## 測試計畫

新增/更新：

- `tests/agent/test_interaction_stance.py`
  - disabled/missing config default
  - status → brief
  - execution → brief
  - exploration → expanded
  - debate → standard
  - route fallback affects density
  - invalid config override ignored
- `tests/agent/test_cognitive_router.py`
  - route carries dialogue fields and old constructor compatibility
- `tests/agent/test_cognition_trace.py`
  - interaction block defaults and populated values
- `tests/agent/test_cognition_trace_report.py`
  - interaction counters
- `tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot`
  - runtime result/metadata includes interaction stance

Targeted test command：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_interaction_stance.py \
  tests/agent/test_cognitive_router.py \
  tests/agent/test_cognition_trace.py \
  tests/agent/test_cognition_trace_report.py \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

## Commit split

1. `docs: plan interaction stance PR10`
2. `feat: add interaction stance planner`
3. `feat: wire interaction stance metadata and reports`

## 安全性 / 相容性

- cognition disabled 仍不破壞既有 run loop。
- 所有新增欄位皆為 metadata/hint，不改 dispatch。
- `CognitiveRoute` 新欄位有 default，舊測試與舊 constructor call 相容。
- trace/report 僅讀 metadata，不改 runtime behavior。
