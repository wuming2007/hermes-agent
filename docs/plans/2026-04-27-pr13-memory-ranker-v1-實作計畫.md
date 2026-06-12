# PR13 Memory Ranker v1 實作計畫

## Base

- Repo worktree: `~/.hermes/hermes-agent/.worktrees/pr13-memory-ranker`
- Branch: `feat/memory-ranker-pr13`
- Base: `runtime/active-cognitive-stack` at `c7c2ef7b`

## 目標

建立第一版 deterministic memory ranking layer，讓 memory retrieval 在進入 prompt 前可以被排序、分層與裁切，作為 PR14 source trace / metadata 的前置基礎。

PR13 不直接改變外部記憶 provider 的儲存格式，不做 learned ranking，不做 online adaptation。先建立穩定、可測、fail-open 的 ranking primitive 與 MemoryManager 接線。

## 非目標

- 不改 MEMORY.md / USER.md schema。
- 不改 session_search 或 Obsidian 檢索。
- 不做 LLM reranker。
- 不做自動 promotion / decay 寫回；那是 PR17。
- 不破壞既有 `prefetch_all()` / `prefetch_for_policy()` 回傳純文字的 contract。

## 設計

### 新模組：`agent/memory_ranker.py`

資料結構：

- `MemoryRankerConfig`
  - `enabled: bool = True`
  - `max_items: int = 8`
  - `max_chars: int = 6000`
  - 權重：`relevance_weight`、`recency_weight`、`importance_weight`、`reinforcement_weight`、`confidence_weight`、`decay_weight`
- `MemoryCandidate`
  - `text: str`
  - `provider: str = ""`
  - `layer: str = "semantic"`
  - `relevance: float = 0.0`
  - `recency: float = 0.0`
  - `importance: float = 0.0`
  - `reinforcement: float = 0.0`
  - `confidence: float = 0.0`
  - `decay_penalty: float = 0.0`
  - `source: str = ""`
  - `metadata: Mapping[str, Any] | None = None`
- `RankedMemory`
  - `candidate: MemoryCandidate`
  - `score: float`
  - `tier: "hot" | "warm" | "cold" | "archive"`
  - `rank: int`

Pure functions：

- `clamp_signal(value) -> float`
- `score_memory_candidate(candidate, config=None) -> float`
- `memory_tier_for_score(score) -> str`
- `rank_memory_candidates(candidates, config=None) -> list[RankedMemory]`
- `build_ranked_memory_context(ranked, config=None) -> str`

行為：

- score deterministic，不呼叫模型、不呼叫工具。
- signal missing/invalid 時以 0 安全處理。
- score 公式：加權正向 signal 減 `decay_weight * decay_penalty`。
- 排序 tie-break：score desc、tier priority、provider、source、text。
- context builder 只輸出文字，不要求下游理解新物件。
- max_items / max_chars 由 config 裁切。

### MemoryProvider backward-compatible hook

在 `agent/memory_provider.py` 增加可選 hook：

```python
def prefetch_candidates(self, query: str, *, layers=(), session_id: str = "") -> list[MemoryCandidate]:
    return []
```

預設回空陣列，避免現有 provider 必須改。

### MemoryManager 新方法

在 `agent/memory_manager.py` 增加：

```python
def prefetch_ranked_for_policy(self, query: str, *, layers, session_id: str = "", ranker_config=None) -> str:
    ...
```

行為：

1. 對每個 provider 呼叫 `prefetch_candidates()`。
2. 收集 candidates 後用 `rank_memory_candidates()` + `build_ranked_memory_context()` 產生 bounded context。
3. 若沒有 provider 提供 candidates，fallback 到現有 `prefetch_for_policy()`。
4. 任一 provider candidate hook 失敗時 fail-open，其他 provider 繼續。
5. ranker 自身失敗時 fallback 到 `prefetch_for_policy()`。

PR13 先不強制 run_agent 改走新方法，除非實作時風險很低；最小可驗收是 MemoryManager 提供可用入口，PR14/後續再把 provider metadata 與 runtime trace 深接。

## 測試計畫（TDD）

新增：`tests/agent/test_memory_ranker.py`

測試：

1. `clamp_signal` 處理 None、字串、負值、超過 1。
2. `score_memory_candidate` 正向 signal 提高分數，decay_penalty 降低分數。
3. `memory_tier_for_score` 分 hot/warm/cold/archive。
4. `rank_memory_candidates` deterministic 排序、rank 從 1 開始、max_items 生效。
5. `build_ranked_memory_context` 包含 tier/score/provider 並遵守 max_chars。
6. disabled config 保留輸入順序但仍產生 bounded ranked wrapper。

擴充：`tests/agent/test_memory_manager.py`

測試：

1. provider candidates path 會產生 ranked context。
2. 無 candidates 時 fallback 到 `prefetch_for_policy()`。
3. provider candidate hook exception non-fatal。
4. ranker exception fallback 不影響舊 retrieval。

測試命令：

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/agent/test_memory_ranker.py tests/agent/test_memory_manager.py -q
```

## Commit 切法

1. `docs: plan memory ranker PR13`
2. `test: cover memory ranker PR13`
3. `feat: add deterministic memory ranker`
4. `feat: wire ranked memory prefetch manager path`

## 驗收標準

- 新增 pure ranker tests 經 RED 後 GREEN。
- MemoryManager ranked path tests 經 RED 後 GREEN。
- 既有 PR2 memory manager tests 不退化。
- 有效修改已 commit。
- 不修改 live runtime checkout，只在 PR13 worktree 開發。
