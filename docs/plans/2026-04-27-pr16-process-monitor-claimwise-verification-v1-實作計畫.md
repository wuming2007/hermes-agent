# PR16 Process Monitor / Claimwise Verification v1 實作計畫

## Base

- Repo worktree: `~/.hermes/hermes-agent/.worktrees/pr16-process-monitor`
- Branch: `feat/process-monitor-pr16`
- Base: `runtime/active-cognitive-stack` at `47e71546`（PR15 已整合）

## 目標

建立第一版 deterministic process monitor / claimwise verification layer，讓 Hermes 在 final consistency guard 之外，能把回答中的關鍵主張拆成 claim，標示是否有 evidence / policy citation / unsupported gap，並把結果寫入 cognition metadata / trace。

PR16 的核心不是讓模型自動改寫答案，而是先建立可觀測、可測、可治理的 claim-level evidence surface，作為後續 PR17 plasticity 與長期 reliability 的基礎。

## 非目標

- 不做 LLM claim extractor。
- 不做多輪 verifier / repair loop。
- 不改 final response 內容。
- 不阻止 tool use 或外部行動；policy enforcement 仍不是本 PR。
- 不把 monitor 結果持久化成 memory；trajectory 已有 cognition_trace 出口。
- 不破壞 PR3 consistency guard dispatch；process monitor 是獨立 metadata layer。

## 新模組：`agent/process_monitor.py`

### Dataclass

```python
@dataclass(frozen=True)
class Claim:
    text: str
    kind: str = "factual"  # factual / action / status / policy / causal / unknown
    evidence_refs: tuple[str, ...] = ()
    policy_refs: tuple[str, ...] = ()
    confidence: str = "unknown"  # high / medium / low / unknown
```

```python
@dataclass(frozen=True)
class ClaimAssessment:
    claim: Claim
    supported: bool
    evidence_gap: bool
    policy_gap: bool
    notes: tuple[str, ...] = ()
    rank: int = 0
```

```python
@dataclass(frozen=True)
class ProcessMonitorReport:
    enabled: bool
    claim_count: int
    supported_count: int
    evidence_gap_count: int
    policy_gap_count: int
    assessments: tuple[ClaimAssessment, ...] = ()
    notes: tuple[str, ...] = ()
```

### Pure functions

- `extract_claims_from_response(response: str, *, max_claims=8) -> list[Claim]`
  - deterministic heuristic。
  - 以句號、換行、bullet 切句。
  - 忽略太短/純寒暄。
  - status/action/factual/policy keyword 粗分 kind。
- `assess_claims(claims, *, evidence_refs=(), policy_refs=()) -> ProcessMonitorReport`
  - claim 若本身或全域 evidence_refs 有值 → supported/evidence_gap false。
  - claim kind 是 `policy` 或 `action` 時，若無 policy_refs → policy_gap true。
  - factual/status/action claim 無 evidence → evidence_gap true。
- `build_process_monitor_metadata(report) -> dict[str, Any]`
  - JSON-friendly：enabled、claim_count、supported_count、evidence_gap_count、policy_gap_count、claim_kinds、unsupported_claims、policy_gap_claims。
- `build_process_monitor_context(report) -> str`
  - compact debug/telemetry string；PR16 不注入 prompt，只供測試與未來 diagnostics。

## Runtime wiring

### `run_agent.py`

在 final response 已確定、consistency guard 已處理後、`build_cognition_turn_trace(...)` 之前：

1. 如果 cognition enabled 且 final_response 非空：
   - 用 `extract_claims_from_response(final_response)`。
   - evidence_refs 來源：
     - `policy_memory_citations` 可算 policy_refs。
     - PR16 暫時沒有 tool evidence registry，因此 evidence_refs 初版可由 `verification_notes` / policy citations / memory citations 轉入最小集合。
   - 用 `assess_claims(...)` 產生 report。
   - 把 `build_process_monitor_metadata(report)` merge 到 `_current_turn_cognition_metadata`。
2. fail-open：process monitor exception 只 log warning/debug，不影響 final_response。
3. cognition disabled 時不跑，trace process 預設 disabled。

新增 metadata keys：

- `process_monitor_enabled`
- `process_monitor_claim_count`
- `process_monitor_supported_count`
- `process_monitor_evidence_gap_count`
- `process_monitor_policy_gap_count`
- `process_monitor_claim_kinds`
- `process_monitor_unsupported_claims`
- `process_monitor_policy_gap_claims`

### `agent/cognition_trace.py`

新增 additive block：

```python
"process_monitor": {
    "enabled": bool,
    "claim_count": int,
    "supported_count": int,
    "evidence_gap_count": int,
    "policy_gap_count": int,
    "claim_kinds": list[str],
    "unsupported_claims": list[str],
    "policy_gap_claims": list[str],
}
```

保持 schema version 1，因為 additive optional field。

## 測試計畫（TDD）

### 新增 `tests/agent/test_process_monitor.py`

1. blank response → disabled/0 claims 或 empty report。
2. extracts bullet/list/sentence claims deterministically。
3. classifies action/status/policy/factual claims。
4. assesses evidence gap for factual/status/action without evidence。
5. policy/action claim without policy_refs gets policy_gap。
6. evidence_refs / policy_refs reduce gaps。
7. metadata is JSON-friendly and contains unsupported/policy-gap claims。
8. context builder includes compact counts。

### 擴充 `tests/agent/test_cognition_trace.py`

- missing process metadata defaults disabled/zero。
- flat process monitor metadata becomes nested `trace["process_monitor"]`。

### 擴充 `tests/run_agent/test_run_agent.py`

新增或擴充 lightweight runtime test：

- routed cognition + final response with factual/action claims produces process monitor metadata and trace block。
- disabled cognition keeps process monitor disabled/absent default。
- monkeypatch process monitor function raising → turn still completes。

## 測試命令

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_process_monitor.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

Full confidence with recent layers:

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_process_monitor.py \
  tests/agent/test_policy_memory.py \
  tests/agent/test_memory_ranker.py \
  tests/agent/test_memory_manager.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q
```

## Commit 切法

1. `docs: plan process monitor PR16`
2. `test: cover process monitor PR16` — RED，應因 `agent.process_monitor` / trace block missing 失敗。
3. `feat: add deterministic process monitor`
4. `feat: wire process monitor metadata into runtime`

## 驗收標準

- Pure functions deterministic，不呼叫模型/工具。
- process monitor fail-open，不影響 final response。
- cognition disabled path 不跑 monitor。
- trace 中可看到 process_monitor block。
- 不改 PR3 consistency guard 的 final response behavior。
- targeted tests 全綠。
- 完成後 commit 並 fast-forward active runtime。
