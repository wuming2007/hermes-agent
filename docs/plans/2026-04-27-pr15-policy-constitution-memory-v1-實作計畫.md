# PR15 Policy / Constitution Memory v1 實作計畫

## Base

- Repo worktree: `~/.hermes/hermes-agent/.worktrees/pr15-policy-memory`
- Branch: `feat/policy-memory-pr15`
- Base: `runtime/active-cognitive-stack` at `2661244a`（PR14 已整合）

## 目標

建立第一版 deterministic policy / constitution memory layer，讓 Hermes 能把「規範、原則、禁止事項、偏好、外部副作用邊界」視為一級可召回記憶，而不是只散落在 system prompt、SOUL.md、skills 或自然語言記憶裡。

PR15 的重點是：

1. 定義穩定的 policy memory object schema。
2. 提供 deterministic policy matching / citation / context builder。
3. 在 MemoryManager ranked prefetch path 中支援 provider 回傳 policy items。
4. 在 cognition metadata / trace 中暴露 policy recall summary，為 PR16 claimwise/process monitor 提供政策依據。

## 非目標

- 不把所有現有 SOUL.md / memory 全自動轉成 policy store。
- 不做 classifier / executable guard enforcement；只做 recall + citation + metadata。
- 不做線上學習或 policy mutation。
- 不改舊 provider 必須實作的介面。
- 不改 cognition disabled / no route 的 legacy `prefetch_all()` path。
- 不把苗苗私人層 upstream 化；schema 保持 persona-neutral。

## 新模組：`agent/policy_memory.py`

### Dataclass

```python
@dataclass(frozen=True)
class PolicyMemoryItem:
    policy_id: str
    title: str
    text: str
    category: str = "general"  # safety / privacy / external_action / memory / style / workflow / general
    priority: int = 50          # 0-100
    scope: str = "global"       # global / user / project / platform
    source_trace: tuple[str, ...] = ()
    version: str = "1"
    enabled: bool = True
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None
```

```python
@dataclass(frozen=True)
class PolicyRecallResult:
    item: PolicyMemoryItem
    score: float
    matched_terms: tuple[str, ...]
    citation: str
    rank: int = 0
```

### Pure functions

- `normalize_policy_memory_item(value) -> PolicyMemoryItem | None`
  - 接受 `PolicyMemoryItem` 或 dict。
  - invalid/missing `policy_id/title/text` 回 `None`。
  - priority clamp 到 0..100。
  - category/scope/tags/source_trace 安全正規化。
- `policy_item_to_candidate(item, query="") -> MemoryCandidate`
  - 轉成 PR13/PR14 `MemoryCandidate`。
  - layer=`principles`
  - provider=`policy`
  - metadata 帶 `MemoryObjectMetadata(source_trace=..., status="active", confidence=1.0)`。
  - source 顯示 `policy:<policy_id>`。
- `score_policy_item(item, query) -> tuple[float, tuple[str, ...]]`
  - deterministic keyword overlap + priority + category boost。
  - 外部副作用關鍵詞：send/email/publish/tweet/delete/寄信/發送/發布/刪除。
  - memory/privacy 關鍵詞：remember/memory/private/資料/記憶/隱私。
- `recall_policy_memories(query, items, max_items=5) -> list[PolicyRecallResult]`
  - disabled item 不回傳。
  - score > 0 才回傳。
  - deterministic sort: score desc, priority desc, policy_id。
- `build_policy_memory_context(results) -> str`
  - compact context，包含 citation、category、priority、source_trace。
- `build_policy_recall_metadata(results) -> dict`
  - JSON-friendly metadata：count、policy_ids、citations、categories。

## MemoryProvider / MemoryManager 接線

### `agent/memory_provider.py`

新增 optional hook：

```python
def prefetch_policy_items(self, query: str, *, session_id: str = "") -> list[Any]:
    return []
```

舊 provider 不需改。

### `agent/memory_manager.py`

新增：

```python
def prefetch_policy_for_query(self, query: str, *, session_id: str = "", max_items: int = 5) -> tuple[str, dict[str, Any]]:
    ...
```

行為：

1. 對 provider 呼叫 `prefetch_policy_items()`。
2. 正規化 policy items。
3. recall + build context + metadata。
4. provider exception non-fatal。
5. 若沒有 policy item，回 `("", {"count": 0, ...})`。

擴充 `prefetch_ranked_for_policy()`：

- 在 normal memory candidates 前，收集 policy items 並轉成 `MemoryCandidate` 合併 ranking。
- principles layer 存在時才拉 policy items；避免 fast/no-principles 未來模式無謂查政策。
- policy recall failure 不影響既有 memory prefetch。

## Runtime metadata / trace wiring

### `run_agent.py`

在已有 cognition route / retrieval policy path 中：

- 呼叫 `prefetch_ranked_for_policy()` 時可讓 manager 更新/回傳 policy metadata？
- 為了最小改動，PR15 先讓 `MemoryManager` 暴露 `last_policy_recall_metadata` property 或回傳可查欄位。
- run_agent 在 prefetch 後把 metadata 加到 `_current_turn_cognition_metadata`：
  - `policy_memory_enabled: bool`
  - `policy_memory_count: int`
  - `policy_memory_ids: list[str]`
  - `policy_memory_citations: list[str]`
  - `policy_memory_categories: list[str]`
- fail-open：任何 policy metadata wiring exception 只 log/debug，不影響 response。

### `agent/cognition_trace.py`

在 nested trace 中新增：

```python
"policy": {
    "enabled": bool,
    "count": int,
    "policy_ids": list[str],
    "citations": list[str],
    "categories": list[str],
}
```

保持 schema_version 1 或升 2？

PR15 選擇保持 `SCHEMA_VERSION = 1`，因為是 additive optional field，不破壞現有 trace readers。若未來有 breaking trace schema 再升版。

## 測試計畫（TDD）

### 新增 `tests/agent/test_policy_memory.py`

1. normalize dict → `PolicyMemoryItem`。
2. invalid item → `None`。
3. score detects external-action query and returns matched terms。
4. recall sorts by score/priority and excludes disabled items。
5. build context includes citation/category/source_trace。
6. item to candidate produces principles-layer `MemoryCandidate` with PR14 metadata。
7. build metadata JSON-friendly。

### 擴充 `tests/agent/test_memory_provider.py`

- default `prefetch_policy_items()` returns `[]`。

### 擴充 `tests/agent/test_memory_manager.py`

1. `prefetch_policy_for_query()` collects provider items and returns context + metadata。
2. provider exception non-fatal。
3. `prefetch_ranked_for_policy()` includes policy candidate when `principles` layer requested。
4. without principles layer, policy hook not called / not included。

### 擴充 `tests/agent/test_cognition_trace.py`

- flat policy metadata becomes nested `trace["policy"]`。
- missing policy metadata defaults to disabled/count 0。

### 擴充 `tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch` 或新增小測

- routed cognition calls ranked prefetch and copies `last_policy_recall_metadata` into result cognition metadata。
- disabled cognition does not add policy metadata beyond defaults。

## 測試命令

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_policy_memory.py \
  tests/agent/test_memory_manager.py \
  tests/agent/test_memory_provider.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

## Commit 切法

1. `docs: plan policy memory PR15`
2. `test: cover policy memory PR15` — RED，應因 `agent.policy_memory` / hooks / trace policy missing 失敗。
3. `feat: add deterministic policy memory layer`
4. `feat: wire policy memory recall metadata`

## 驗收標準

- Policy memory pure functions deterministic，不呼叫模型/工具。
- 舊 provider 不需改，default hook 保持 no-op。
- policy recall 只 additive，不阻斷舊 ranked memory retrieval。
- cognition trace 中可看到 policy recall summary。
- invalid policy item/provider exception fail-open。
- targeted tests 全綠。
- 完成後 commit 並 fast-forward active runtime。
