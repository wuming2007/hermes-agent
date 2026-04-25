# PR9：Cognition Trace Offline Report 實作計畫

> **For Hermes:** 以 chained worktree 小 PR 方式實作；先純分析器、再 CLI/script 入口。嚴格 TDD，保持 runtime 無行為變更。

**Goal:** 從 PR8 trajectory JSONL 的 `metadata.cognition_trace` 離線產生 deterministic cognition trace 統計報告，作為未來校準/觀測的資料基礎。

**Architecture:** 新增 `agent.cognition_trace_report` 純分析模組，讀取 trajectory JSONL entries 或檔案路徑，統計 trace schema / route / uncertainty / verification 的分布與 malformed/missing 狀態。再新增 `scripts/cognition_trace_report.py` 作為離線 CLI，輸出 JSON，且不接 runtime loop、不改 policy、不寫 memory。

**Tech Stack:** Python stdlib (`json`, `argparse`, `dataclasses`, `collections.Counter`, `pathlib`) + pytest。

---

## Base / Branch / Worktree

- Base branch: `feat/cognition-trace-trajectory-pr8`
- PR9 branch: `feat/cognition-trace-report-pr9`
- Worktree: `/Users/wuming/.hermes/hermes-agent/.worktrees/pr9-cognition-trace-report`
- Plan file: `docs/plans/2026-04-25-pr9-cognition-trace-report-實作計畫.md`

## 非目標

- 不做 online adaptation / learned calibration。
- 不改 runtime routing、uncertainty policy、verification ladder 或 model dispatch。
- 不改 memory schema，不寫入長期記憶。
- 不上傳 telemetry，不新增背景 job。
- 不改 trajectory conversation format；只讀 PR8 已存在的 optional `metadata.cognition_trace`。

## 輸出報告 shape（穩定 JSON-friendly）

```json
{
  "schema_version": 1,
  "files": ["trajectory_samples.jsonl"],
  "total_entries": 0,
  "completed": {"true": 0, "false": 0, "missing": 0},
  "cognition_trace": {
    "present": 0,
    "missing": 0,
    "enabled": {"true": 0, "false": 0, "missing": 0},
    "schema_versions": {},
    "malformed": 0
  },
  "route": {
    "modes": {},
    "original_modes": {},
    "retrieval_plans": {},
    "verification_plans": {},
    "allow_cheap_model": {"true": 0, "false": 0, "missing": 0},
    "consistency_check": {"true": 0, "false": 0, "missing": 0}
  },
  "uncertainty": {
    "present": {"true": 0, "false": 0, "missing": 0},
    "confidence_bands": {},
    "actions": {},
    "depth_escalated": {"true": 0, "false": 0, "missing": 0},
    "require_tool_evidence": {"true": 0, "false": 0, "missing": 0},
    "seek_human": {"true": 0, "false": 0, "missing": 0},
    "target_modes": {}
  },
  "verification": {
    "ladder_enabled": {"true": 0, "false": 0, "missing": 0},
    "ladder_source_plans": {},
    "ladder_stages": {},
    "ladder_applied_stages": {},
    "applied": {"true": 0, "false": 0, "missing": 0},
    "changed": {"true": 0, "false": 0, "missing": 0}
  },
  "errors": {"malformed_jsonl": 0, "missing_files": 0}
}
```

## Task 1：純離線分析器 + unit tests

**Objective:** 新增 deterministic pure analyzer，可以分析 in-memory entries 與 JSONL 檔案。

**Files:**
- Create: `agent/cognition_trace_report.py`
- Create: `tests/agent/test_cognition_trace_report.py`

**TDD steps:**
1. 先寫 tests：
   - empty input returns zero report shape。
   - counts completed true/false/missing。
   - counts present/missing/malformed `metadata.cognition_trace`。
   - counts route mode / original mode / retrieval / verification plans。
   - counts uncertainty action/confidence/booleans/target modes。
   - counts verification ladder source/stages/applied stages。
   - JSONL reader skips malformed lines and increments `errors.malformed_jsonl`。
   - missing file increments `errors.missing_files` and does not raise。
2. 跑 RED：
   ```bash
   HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/agent/test_cognition_trace_report.py -q
   ```
3. 實作 minimal module：
   - `TRACE_REPORT_SCHEMA_VERSION = 1`
   - `empty_cognition_trace_report(files: Sequence[str] | None = None) -> dict[str, Any]`
   - `analyze_cognition_trace_entries(entries: Iterable[Mapping[str, Any]], files: Sequence[str] | None = None) -> dict[str, Any]`
   - `analyze_cognition_trace_jsonl(paths: Iterable[str | Path]) -> dict[str, Any]`
   - helper functions normalize scalar keys with `str(value)` and booleans into `true/false/missing` buckets。
4. 跑 GREEN targeted tests。
5. Commit:
   ```bash
   git add agent/cognition_trace_report.py tests/agent/test_cognition_trace_report.py
   git commit -m "feat: add offline cognition trace report analyzer"
   ```

## Task 2：CLI/script 入口 + tests

**Objective:** 提供離線命令列入口，讀多個 JSONL 檔並輸出 report JSON。

**Files:**
- Create: `scripts/cognition_trace_report.py`
- Create: `tests/scripts/test_cognition_trace_report_cli.py`

**TDD steps:**
1. 先寫 CLI tests：
   - `main([path])` prints pretty JSON by default。
   - `main([path, "--compact"])` prints compact JSON。
   - missing file does not crash; output contains `errors.missing_files == 1`。
2. 跑 RED：
   ```bash
   HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' tests/scripts/test_cognition_trace_report_cli.py -q
   ```
3. 實作 `scripts/cognition_trace_report.py`：
   - argparse positional `paths`。
   - `--compact` toggles JSON separators/indent。
   - `main(argv: Sequence[str] | None = None) -> int`。
   - `if __name__ == "__main__": raise SystemExit(main())`。
4. 跑 GREEN CLI tests。
5. Commit:
   ```bash
   git add scripts/cognition_trace_report.py tests/scripts/test_cognition_trace_report_cli.py
   git commit -m "feat: add cognition trace report CLI"
   ```

## Final verification

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_cognition_trace_report.py \
  tests/scripts/test_cognition_trace_report_cli.py \
  tests/agent/test_trajectory_metadata.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestCognitionTraceTrajectoryExport \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  -q

git status --short --branch
git log --oneline --decorate -8
```

## Safety notes

- Analyzer is offline-only and deterministic。
- Missing/malformed input is counted in report, not raised as runtime error。
- No existing runtime entrypoint imports this script, so disabled cognition/runtime behavior remains unchanged。
- Report fields are JSON-friendly for downstream calibration notebooks or future PRs。
