# PR18 Autonomy / Self-model Telemetry v1 實作計畫

> **For Hermes:** Follow TDD. Keep this PR deterministic, metadata-only, fail-open, and additive to the PR13–PR17 cognition stack.

**Goal:** 建立自律程度與自我模型遙測 v1，讓每回合可以以可稽核的方式記錄「能力、介入/核准、外部動作邊界、證據需求」等訊號，作為未來更強 autonomy policy 的基礎。

**Architecture:** 新增 pure module `agent/autonomy_telemetry.py`，從既有 cognition metadata、tool/action hints、policy/process/plasticity signals 產生 bounded JSON-friendly telemetry。runtime 只把 metadata 合併進 `_current_turn_cognition_metadata` 與 `cognition_trace`；不改模型輸出、不自動升權、不自動執行外部動作。

**Tech Stack:** Python dataclasses + pure functions + pytest；Hermes runtime shared venv `~/.hermes/hermes-agent/venv/bin/python`。

---

## Base

- Repo: `~/.hermes/hermes-agent`
- Worktree: `~/.hermes/hermes-agent/.worktrees/pr18-autonomy-telemetry`
- Branch: `feat/autonomy-self-model-telemetry-pr18`
- Base: `runtime/active-cognitive-stack` at PR17 commit `6b09f8bf`

## Non-goals

- 不新增長期記憶寫入。
- 不做 learned autonomy calibration。
- 不改 tool approval / dangerous command policy。
- 不改 final response。
- 不把 self-model 當人格宣告；v1 只做遙測觀測。
- telemetry failure 必須 fail-open，不影響 user turn。

## New module

Create `agent/autonomy_telemetry.py` with frozen dataclasses:

- `AutonomySignal`
  - `requested_action: str = ""`
  - `external_action: bool = False`
  - `user_approval_present: bool = False`
  - `tool_evidence_present: bool = False`
  - `policy_support_present: bool = False`
  - `process_evidence_gap_count: int = 0`
  - `process_policy_gap_count: int = 0`
  - `plasticity_promoted_count: int = 0`
  - `plasticity_decayed_count: int = 0`
  - `competence_band: str = "unknown"`
  - `risk_level: str = "low"`

- `AutonomyTelemetry`
  - `enabled: bool`
  - `autonomy_level: str` — `observe | assist | act_with_approval | blocked_pending_evidence`
  - `competence_band: str`
  - `risk_level: str`
  - `external_action: bool`
  - `approval_required: bool`
  - `approval_present: bool`
  - `evidence_required: bool`
  - `evidence_present: bool`
  - `policy_supported: bool`
  - `intervention_reasons: tuple[str, ...]`
  - `self_model_notes: tuple[str, ...]`

Pure helpers:

- `normalize_autonomy_signal(value) -> AutonomySignal`
- `resolve_autonomy_telemetry(signal, metadata=None) -> AutonomyTelemetry`
- `build_autonomy_metadata(telemetry) -> dict[str, Any]`
- `build_autonomy_context(telemetry) -> str`
- `build_autonomy_telemetry_from_metadata(metadata) -> AutonomyTelemetry`

## Deterministic rules

- `external_action=True` always requires approval.
- `risk_level in {medium, high}` requires evidence.
- existing `require_tool_evidence=True`, process evidence gaps, or policy gaps require evidence.
- no approval for external action => `blocked_pending_evidence` or `act_with_approval` depending approval/evidence state.
- internal low-risk work => `assist`.
- empty/no metadata => disabled telemetry with `autonomy_level="observe"`.
- input coercion must be bounded and JSON-friendly.

## Runtime wiring

Modify `run_agent.py` after process monitor metadata and before cognition trace construction:

1. Import `build_autonomy_metadata` and `build_autonomy_telemetry_from_metadata`.
2. If `_current_cognitive_route is not None` and `_current_turn_cognition_metadata` is a dict:
   - build telemetry from metadata.
   - update `_current_turn_cognition_metadata` with flat autonomy keys.
3. Wrap in separate `try/except Exception` and log/debug only; final response unchanged.

Modify `agent/cognition_trace.py`:

Add sibling block:

```python
"autonomy": {
    "enabled": bool,
    "level": str,
    "competence_band": str,
    "risk_level": str,
    "external_action": bool,
    "approval_required": bool,
    "approval_present": bool,
    "evidence_required": bool,
    "evidence_present": bool,
    "policy_supported": bool,
    "intervention_reasons": list[str],
    "self_model_notes": list[str],
}
```

## Tests first

Create `tests/agent/test_autonomy_telemetry.py` covering:

1. normalization clamps integers, booleans, allowed bands/levels.
2. empty signal resolves disabled observe telemetry.
3. internal low-risk metadata resolves assist with no approval required.
4. external action without approval/evidence records required approval/evidence and intervention reasons.
5. policy support/tool evidence can satisfy action telemetry.
6. metadata builder emits flat `autonomy_*` keys.
7. context builder renders compact readable summary.

Extend `tests/agent/test_cognition_trace.py`:

- default absent autonomy block.
- populated autonomy metadata grouped into `trace["autonomy"]`.

Extend `tests/run_agent/test_run_agent.py` with a small runtime wiring class if existing helpers allow:

- telemetry metadata appears in `cognition_metadata` / `cognition_trace` for routed cognition.
- fail-open if `run_agent.build_autonomy_telemetry_from_metadata` raises.

## RED command

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_autonomy_telemetry.py \
  tests/agent/test_cognition_trace.py::test_autonomy_defaults_when_absent \
  tests/agent/test_cognition_trace.py::test_autonomy_metadata_is_grouped \
  -q
```

Expected initial RED: `ModuleNotFoundError: No module named 'agent.autonomy_telemetry'` or missing trace block.

## GREEN / verification command

```bash
HERMES_DISABLE_STDERR_NOISE_FILTER=1 ~/.hermes/hermes-agent/venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_autonomy_telemetry.py \
  tests/agent/test_memory_plasticity.py \
  tests/agent/test_process_monitor.py \
  tests/agent/test_policy_memory.py \
  tests/agent/test_memory_ranker.py \
  tests/agent/test_memory_manager.py \
  tests/agent/test_memory_provider.py \
  tests/agent/test_cognition_trace.py \
  tests/run_agent/test_run_agent.py::TestLayeredRetrievalPrefetch \
  tests/run_agent/test_run_agent.py::TestCognitiveRouting \
  tests/run_agent/test_run_agent.py::TestCognitionTurnMetadataSnapshot \
  tests/run_agent/test_run_agent.py::TestProcessMonitorWiring \
  -q
```

## Commit split

1. `docs: plan autonomy telemetry PR18`
2. `test: cover autonomy telemetry PR18`
3. `feat: add deterministic autonomy telemetry`

## Acceptance

- All new tests pass.
- PR13–PR18 confidence suite passes.
- `agent.autonomy_telemetry` import smoke passes.
- Feature branch fast-forward merges into `runtime/active-cognitive-stack`.
- Active runtime HEAD equals PR18 implementation commit.
