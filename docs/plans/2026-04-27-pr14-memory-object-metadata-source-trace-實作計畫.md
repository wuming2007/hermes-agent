# PR14 Memory Object Metadata / Source Trace 實作計畫

## Base

- Repo worktree: `~/.hermes/hermes-agent/.worktrees/pr14-memory-object-metadata`
- Branch: `feat/memory-object-metadata-pr14`
- Base: `runtime/active-cognitive-stack` at `7eeb4fdc`（PR13 已整合）

## 目標

在 PR13 `MemoryCandidate` / ranked prefetch 的基礎上，加入第一版可追溯、可審計的 memory object metadata。

PR14 的重點不是改變所有記憶 backend 的儲存格式，而是先提供穩定的 metadata schema、正規化函式、context 呈現與 manager path，讓未來 PR15 policy memory、PR16 claimwise verification、PR17 plasticity 可以安全引用記憶來源與狀態。

## 非目標

- 不修改 `MEMORY.md` / `USER.md` 檔案格式。
- 不要求既有 provider 必須立即產生 metadata。
- 不做資料庫 migration。
- 不做 learned source confidence。
- 不自動 promotion / decay / supersession 寫回；那是 PR17。
- 不改變 cognition disabled 時的 legacy `prefetch_all()` path。

## 新增 / 擴充設計

### `agent/memory_ranker.py`

新增 frozen dataclass：

```python
@dataclass(frozen=True)
class MemoryObjectMetadata:
    source_trace: tuple[str, ...] = ()
    compression_level: int = 0
    confidence: float | None = None
    last_verified_at: str = ""
    superseded_by: str = ""
    reinforcement_count: int = 0
    status: str = "active"  # active / stale / superseded / inferred / unverified
    notes: Mapping[str, Any] | None = None
```

新增 pure functions：

- `normalize_memory_metadata(value) -> MemoryObjectMetadata`
  - 接受 `None`、`MemoryObjectMetadata`、dict、其他 invalid value。
  - dict 中的 `source_trace` 可為 string/list/tuple；輸出一律 tuple[str, ...]。
  - `compression_level`、`reinforcement_count` 安全轉 int 且不小於 0。
  - `confidence` 使用 `clamp_signal`，但 missing 時保持 `None`。
  - `status` 僅允許 `active/stale/superseded/inferred/unverified`，未知值轉 `unverified`。
- `memory_metadata_to_dict(metadata) -> dict[str, Any]`
  - JSON-friendly，不輸出 tuple。
- `candidate_with_normalized_metadata(candidate) -> MemoryCandidate`
  - 不 mutation，回傳 metadata 已正規化的候選。
- `memory_metadata_label(metadata) -> str`
  - 產生 compact label，例如 `status=active confidence=0.80 verified=2026-04-27 source=MEMORY.md>USER.md compression=1 reinforced=3`。

擴充 `MemoryCandidate`：

- 保留 `metadata: Mapping[str, Any] | MemoryObjectMetadata | None = None`，不破壞舊呼叫。
- 新增 optional `object_id: str = ""`，用於 `superseded_by` 或 provider 自己的 id 對照。

擴充 `build_ranked_memory_context(...)`：

- 若 candidate metadata 可正規化，header 中加入：
  - `status=...`
  - `confidence=...`
  - `verified=...`
  - `source_trace=...`
  - `superseded_by=...`
  - `compression=...`
  - `reinforced=...`
- 若 metadata invalid，不拋錯；視為 `unverified`。
- 舊 candidate 沒 metadata 時仍正常輸出。

### `agent/memory_provider.py`

新增 optional provider hook：

```python
def describe_memory_object_metadata(self) -> Mapping[str, Any]:
    return {}
```

用途：provider 可宣告自己支援哪些 metadata keys，供 manager/system prompt 或後續 diagnostics 使用。PR14 先只提供 hook，不強制 runtime 使用。

### `agent/memory_manager.py`

新增：

```python
def describe_memory_metadata_support(self) -> dict[str, Any]:
    ...
```

行為：

- 對每個 provider 呼叫 `describe_memory_object_metadata()`。
- provider exception non-fatal。
- 回傳 JSON-friendly dict，例如：
  `{"providers": {"builtin": {...}, "honcho": {...}}}`。

`prefetch_ranked_for_policy()` 在收集 candidates 後，先用 `candidate_with_normalized_metadata()` 正規化，再排名與建 context。若正規化失敗，fail-open：保留原 candidate 或 fallback，不影響 turn。

## 測試計畫（TDD）

### 新增 / 擴充 `tests/agent/test_memory_ranker.py`

1. `normalize_memory_metadata(None)` 產生 unverified/default metadata。
2. dict metadata 可正規化：source_trace、compression_level、confidence、last_verified_at、superseded_by、reinforcement_count、status。
3. invalid metadata 不拋錯，輸出 unverified/default。
4. `memory_metadata_to_dict()` JSON-friendly，source_trace 是 list。
5. `candidate_with_normalized_metadata()` 不 mutate 原 candidate。
6. `build_ranked_memory_context()` header 顯示 metadata label。
7. superseded/stale candidate 可被看到，不被靜默丟棄；真正降權仍留給 PR17。

### 擴充 `tests/agent/test_memory_manager.py`

1. ranked prefetch 會正規化 provider candidate metadata，context 中可看到 source/confidence/status。
2. provider metadata support description 被 manager 彙整。
3. provider description exception non-fatal。

### 可能新增 `tests/agent/test_memory_provider.py`

- default `describe_memory_object_metadata()` 回 `{}`，舊 provider 無需修改。

## 測試命令

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_memory_ranker.py \
  tests/agent/test_memory_manager.py \
  tests/agent/test_memory_provider.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

## Commit 切法

1. `docs: plan memory object metadata PR14`
2. `test: cover memory object metadata PR14` — RED，應因 import/function missing 失敗。
3. `feat: add memory object metadata schema`
4. `feat: wire memory metadata support through manager`

## 驗收標準

- 新 metadata functions 全部 pure / deterministic。
- 舊 `MemoryCandidate` 呼叫仍通過。
- invalid/missing metadata 不會讓 runtime retrieval 失敗。
- cognition route 有 ranked prefetch 時，metadata 可出現在 injected context header。
- cognition disabled / no route path 不變。
- targeted tests 全綠。
- 有效修改完成後 commit，並 fast-forward active runtime。
