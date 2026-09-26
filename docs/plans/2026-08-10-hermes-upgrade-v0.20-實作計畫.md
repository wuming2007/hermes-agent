# Hermes 升級 v0.20.0 (2026.8.3) — cognitive stack 遷移實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 cognitive stack（PR1–18）從 `feat/cognitive-stack-v0.16`（基底 v2026.6.5）遷移到 upstream `v2026.8.3`（= v0.20.0），產出可上 production 的 `feat/cognitive-stack-v0.20` 分支，完成 runtime handoff。

**Architecture:** 沿用「重建而非 rebase」playbook：從 upstream tag 開新分支，Cat 1 新檔 verbatim copy、共同檔案 3-way 套用（本次 5 檔可全自動、4 檔小衝突手動解）、8 個整合 hook 中 1–4 隨 3-way 自動帶過，5/6 重接到新模組 `agent/turn_context.py`、7/8 重接到新模組 `agent/turn_finalizer.py`，cognition config block 移到新家 `hermes_cli/config_defaults.py`。`agent/conversation_loop.py` 本次**直接用 upstream 原版**（我們的修改全數搬遷）。

**Tech Stack:** Python 3.11–3.13（**不可 3.14**，`requires-python = ">=3.11,<3.14"` 不變）、uv、pytest。pydantic 維持 2.13.4。

## Global Constraints

- 每個 task 結束必跑 import smoke，失敗不得 commit。
- 修測試不修產品碼：測試失敗時先判斷是「測試引用舊模組位置」還是「產品行為真的壞了」；只有前者可改測試。
- 行號規約：「舊」= `c70037042`（`feat/cognitive-stack-v0.16` tip；程式碼與 `7eab08218` 完全相同，僅多一個 docs commit）；「新」= `v2026.8.3` 各檔。所有舊行號已於 2026-08-10 驗證。
- `.upgrade-progress.md` 每個 task 完成後附加一段紀錄（task 名、結果、測試數字）。
- Commit message 一律附 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

## 背景與調查結論（2026-08-10 調查，全部已驗證）

### 版本脈絡

| 項目 | 值 |
|---|---|
| 目前分支 | `feat/cognitive-stack-v0.16`（tip = `c70037042` 之後僅 docs/progress commits；**程式碼 delta 基準固定用 `c70037042`**） |
| 目前基底 | `3c231eb39` = upstream v0.16.0（tag `v2026.6.5`） |
| 本地 delta | `git diff v2026.6.5..c70037042`：62+ 檔、+11.5k 行 |
| 升級目標 | tag `v2026.8.3` = v0.20.0 |
| upstream 差距 | 9,860 commits / 6,865 檔 |
| runtime 工作樹 | `/Users/wuming/.hermes/hermes-agent`，branch `cognitive-stack-v0.16`，乾淨（僅 untracked `trajectory_samples.jsonl`） |

### merge-tree 試合併結果（base=v2026.6.5, ours=c70037042, theirs=v2026.8.3）

| 結果 | 檔案 |
|---|---|
| **自動合併乾淨（5 檔）** | `agent/agent_init.py`（hook 2 存活，位於合併後 ~1837 行）、`agent/memory_provider.py`、`agent/trajectory.py`、`run_agent.py`（hook 1/3/4 存活）、`tests/agent/test_memory_provider.py` |
| **內容衝突（6 檔）** | `agent/conversation_loop.py`（3 hunks）、`agent/memory_manager.py`（2 hunks，均為雙方各自附加）、`hermes_cli/config.py`（1 hunk：DEFAULT_CONFIG 整個搬走）、`hermes_logging.py`（1 hunk：僅 `import queue`）、`tests/test_hermes_logging.py`（1 hunk：fixture 雙方各自附加）、`tests/run_agent/test_run_agent.py`（1 hunk：theirs 側為空，保留 ours 即可） |

### 三大結構搬家（衝突的真正原因）

1. **turn 前置邏輯 → `agent/turn_context.py`（新檔）**：`on_turn_start` 通知與 `prefetch_all` 區塊從 `run_conversation` 抽到 `build_turn_context()`（v2026.8.3 該檔 ~1147–1166 行），`run_conversation` 改為消費 `_ctx.ext_prefetch_cache`。upstream 並新加 `is_trivial_prompt` 閘門（瑣碎輸入不 prefetch）。→ **hook 5/6 移到這裡**。
2. **turn 收尾邏輯 → `agent/turn_finalizer.py`（新檔）**：`completed` 判定、trajectory 儲存、result dict 組裝全部抽到 `finalize_turn()`（`completed = (` 在 ~195、save-trajectory 在 ~247、`"cost_source"` 在 ~658）。`logger` 在函式內 lazy import 自 `agent.conversation_loop`，logger 名不變。→ **hook 7/8 與 cognition imports 移到這裡**。
3. **DEFAULT_CONFIG → `hermes_cli/config_defaults.py`（新檔）**：`config.py` 只剩 `from hermes_cli.config_defaults import DEFAULT_CONFIG, OPTIONAL_ENV_VARS`。`_FALLBACK_COMMENT` 仍在 `config.py`（~3448 行）。→ **cognition block 移到 config_defaults.py；YAML 註解照舊附在 _FALLBACK_COMMENT 尾端**。

### Hook 對應表（8 hooks → v0.20 位置）

| # | 內容 | 舊位置（c70037042） | v0.20 處置 |
|---|---|---|---|
| 1 | cognition 模組 imports | conversation_loop.py:54–71 | 移到 `agent/turn_finalizer.py` import 區（消費端 hook 7/8 在那裡）；conversation_loop **零新增** |
| 2 | `__init__` cognition config 初始化 | agent_init.py（既有） | **3-way 自動帶過**（Task 3） |
| 3 | `_save_trajectory` 帶 metadata | run_agent.py（既有） | **3-way 自動帶過**（Task 3）；`finalize_turn` 仍呼叫 `agent._save_trajectory` |
| 4 | `_resolve_current_cognitive_route` + `_default_consistency_verifier` | run_agent.py（既有） | **3-way 自動帶過**（Task 3） |
| 5 | 每回合 route resolve | conversation_loop.py:781–788 | `agent/turn_context.py` `build_turn_context()` 內、`# Notify memory providers` 註解之前（新 ~1146）。錨點處 `messages`、`original_user_message` 均已綁定（已驗證） |
| 6 | PR2 layer-aware prefetch | conversation_loop.py:800–852 | 取代 `turn_context.py` 的 prefetch 區塊（新 ~1155–1166）；變數 `_ext_prefetch_cache` → `ext_prefetch_cache`；保留 `is_trivial_prompt` 閘門包住兩條路徑 |
| 7 | post-generation pipeline（PR3/16/18/7，158 行） | conversation_loop.py:4657–4814 | `agent/turn_finalizer.py` `finalize_turn()` 內、`# Post-loop cleanup must never lose the response.` 註解之前（新 ~233）。縮排同為 4 空格、已是 `agent.` 形態，**免改寫** |
| 8 | result dict 加 cognition 欄位 | conversation_loop.py:5038–5041 | `turn_finalizer.py` result dict `"cost_source": agent.session_cost_source,`（新 ~658）之後 |

### 已驗證的關鍵事實

- Cat 1 新增檔在 v2026.8.3 **零同名衝突**。
- 外部依賴存活：`tools.registry.tool_error`（v2026.8.3:tools/registry.py:930）、`agent.auxiliary_client.get_text_auxiliary_client`（:6451）、`providers.<id>.stale_timeout_seconds` 讀取路徑存在。
- `finalize_turn()` 內無 `_verification_ladder_plan` / `_cognition_trace` 名稱衝突；`final_response` / `interrupted` / `messages` / `completed` 均在作用域。
- `is_trivial_prompt` 由 `agent.memory_provider` 提供，turn_context.py:44 已 import（我們的 memory_provider delta 自動合併，不影響）。
- `f11cfddef`（hermes_logging pytest guard）upstream 無等效修正，隨 3-way 自動帶過（衝突僅 `import queue` 一處）。
- 測試 caplog 斷言 `logger="agent.conversation_loop"` **不用改**（finalize_turn lazy import 保留 logger 名）。
- upstream v0.20 自帶「verification gate」機制（`_pending_verification_response` 等）——與 PR3 guard 屬不同層（它在主迴圈內扣住候選回覆；我們在 finalize 階段複核 final_response），理論上可組合；列入觀察期注意事項。

### 檔案分類總表

| 類別 | 檔案 | 處理 |
|---|---|---|
| Cat 1 verbatim | 14 個 `agent/` cognitive 模組、`scripts/cognition_trace_report.py`、19 個測試檔、`docs/plans/`、`.upgrade-progress.md`、`.upgrade-progress.ts` | Task 2 `git checkout` |
| Cat 2 乾淨 3-way | `agent/agent_init.py`、`agent/memory_provider.py`、`agent/trajectory.py`、`run_agent.py`、`tests/agent/test_memory_provider.py` | Task 3 |
| Cat 2 衝突手解 | `agent/memory_manager.py`、`hermes_logging.py`、`tests/test_hermes_logging.py` | Task 4 |
| 特殊：config | `hermes_cli/config.py`（用 upstream）+ `hermes_cli/config_defaults.py`（塞 cognition block）+ `_FALLBACK_COMMENT` 尾端註解 | Task 5 |
| 特殊：hook 搬遷 | `agent/conversation_loop.py`（用 upstream 原版）、`agent/turn_context.py`（hook 5/6）、`agent/turn_finalizer.py`（hook 7/8 + imports） | Task 6、7 |
| 測試搬遷 | `tests/run_agent/test_run_agent.py`（1 hunk + mock path / getsource 重指向） | Task 8 |

---

## Task 0: 前置檢查與安全網

**Files:** 無程式碼變動（tag + progress log）。

- [ ] **Step 1: 確認工作樹乾淨、tag 與 tip 正確**

```bash
cd /Volumes/Data/openclaw/workspace/hermes-agent
git status --porcelain          # 預期：空
git rev-parse v2026.8.3         # 預期：存在（已 fetch 過）
git merge-base --is-ancestor c70037042 HEAD && echo lineage-OK   # 預期：lineage-OK（c70037042 之後僅 docs/progress commits）
```

- [ ] **Step 2: 記錄現分支 cognitive 測試基準（遷移後全綠標準）**

```bash
uv run pytest tests/agent/test_cognitive_router.py tests/agent/test_cognition_config.py \
  tests/agent/test_consistency_guard.py tests/agent/test_uncertainty_policy.py \
  tests/agent/test_retrieval_policy.py tests/agent/test_cognition_trace.py \
  tests/agent/test_cognition_trace_report.py tests/agent/test_cognition_observation_benchmark.py \
  tests/agent/test_process_monitor.py tests/agent/test_autonomy_telemetry.py \
  tests/agent/test_memory_ranker.py tests/agent/test_interaction_stance.py \
  tests/agent/test_memory_manager.py tests/agent/test_memory_plasticity.py \
  tests/agent/test_policy_memory.py tests/agent/test_trajectory_metadata.py \
  tests/hermes_cli/test_config_cognition_defaults.py \
  tests/run_agent/test_cognition_observation_benchmark.py \
  tests/scripts/test_cognition_trace_report_cli.py -q 2>&1 | tail -3
```

把「X passed」數字記下來（下方 Step 4 寫入 progress log；Task 10 以此為底線）。

- [ ] **Step 3: 打安全 tag**

```bash
TS=$(date +%Y%m%d-%H%M)
git tag "pre-upgrade-$TS/feat/cognitive-stack-v0.16" feat/cognitive-stack-v0.16
git tag "pre-upgrade-$TS/main" main
git push origin "pre-upgrade-$TS/feat/cognitive-stack-v0.16"
echo "$TS" > .upgrade-progress.ts
```

- [ ] **Step 4: `.upgrade-progress.md` 開新章節（附加，勿覆蓋）**

```markdown
# ─────────────────────────────────────────────
# Hermes upgrade to v0.20.0 — progress log
Started: <當下時間>
Target:  v2026.8.3 (v0.20.0, 2026-08-03)
Plan:    docs/plans/2026-08-10-hermes-upgrade-v0.20-實作計畫.md
Spec:    docs/plans/2026-08-10-hermes-upgrade-v0.20-設計規格.md
Baseline cognitive tests on c70037042: <X> passed
Rollback tag prefix: pre-upgrade-<TS>/...
```

- [ ] **Step 5: Commit（在舊分支上）**

```bash
git add .upgrade-progress.md && git commit -m "docs(upgrade): open v0.20.0 migration progress log"
```

## Task 1: 建立新分支與環境

**Files:** 無（分支 + 依賴）。

- [ ] **Step 1: 開分支**

```bash
git switch -c feat/cognitive-stack-v0.20 v2026.8.3
```

- [ ] **Step 2: 同步依賴**

```bash
python3 --version          # 必須 3.11–3.13
uv sync 2>&1 | tail -5     # 預期成功
```

- [ ] **Step 3: upstream 裸基底 import smoke（含兩個新模組）**

```bash
uv run python -c "import run_agent; import agent.conversation_loop, agent.agent_init, agent.turn_context, agent.turn_finalizer; print('baseline OK')"
```

預期 `baseline OK`。

## Task 2: Cat 1 — verbatim copy 新檔案

**Files:** Create（自舊分支整檔取回）：14 個 agent 模組 + script + 19 個測試檔 + docs/plans + progress log。

**Interfaces:**
- Produces: `agent.cognitive_router.resolve_cognitive_route`、`agent.cognition_config.get_cognition_config`、`agent.retrieval_policy.resolve_retrieval_policy`、`agent.consistency_guard.*`、`agent.process_monitor.*`、`agent.autonomy_telemetry.*`、`agent.cognition_trace.build_cognition_turn_trace` 等——Task 3–8 重接的 hook 全部消費這些模組。

- [ ] **Step 1: 取回檔案**

```bash
git checkout feat/cognitive-stack-v0.16 -- \
  agent/autonomy_telemetry.py agent/cognition_config.py \
  agent/cognition_observation_benchmark.py agent/cognition_trace.py \
  agent/cognition_trace_report.py agent/cognitive_router.py \
  agent/consistency_guard.py agent/interaction_stance.py \
  agent/memory_plasticity.py agent/memory_ranker.py \
  agent/policy_memory.py agent/process_monitor.py \
  agent/retrieval_policy.py agent/uncertainty_policy.py \
  scripts/cognition_trace_report.py \
  tests/agent/test_autonomy_telemetry.py tests/agent/test_cognition_config.py \
  tests/agent/test_cognition_observation_benchmark.py tests/agent/test_cognition_trace.py \
  tests/agent/test_cognition_trace_report.py tests/agent/test_cognitive_router.py \
  tests/agent/test_consistency_guard.py tests/agent/test_interaction_stance.py \
  tests/agent/test_memory_manager.py tests/agent/test_memory_plasticity.py \
  tests/agent/test_memory_ranker.py tests/agent/test_policy_memory.py \
  tests/agent/test_process_monitor.py tests/agent/test_retrieval_policy.py \
  tests/agent/test_trajectory_metadata.py tests/agent/test_uncertainty_policy.py \
  tests/hermes_cli/test_config_cognition_defaults.py \
  tests/run_agent/test_cognition_observation_benchmark.py \
  tests/scripts/test_cognition_trace_report_cli.py \
  docs/plans .upgrade-progress.md .upgrade-progress.ts
```

- [ ] **Step 2: 14 個模組 import smoke**

```bash
uv run python -c "
import agent.cognitive_router, agent.cognition_config, agent.cognition_trace, \
       agent.cognition_trace_report, agent.cognition_observation_benchmark, \
       agent.consistency_guard, agent.uncertainty_policy, agent.retrieval_policy, \
       agent.process_monitor, agent.autonomy_telemetry, agent.memory_ranker, \
       agent.memory_plasticity, agent.policy_memory, agent.interaction_stance
print('cognitive modules OK')"
```

- [ ] **Step 3: 跑純模組測試**

```bash
uv run pytest tests/agent/test_cognitive_router.py tests/agent/test_cognition_config.py \
  tests/agent/test_consistency_guard.py tests/agent/test_uncertainty_policy.py \
  tests/agent/test_retrieval_policy.py tests/agent/test_process_monitor.py \
  tests/agent/test_autonomy_telemetry.py tests/agent/test_memory_ranker.py \
  tests/agent/test_memory_plasticity.py tests/agent/test_policy_memory.py \
  tests/agent/test_interaction_stance.py tests/agent/test_cognition_trace.py \
  tests/agent/test_cognition_trace_report.py -q 2>&1 | tail -3
```

預期全綠。`test_cognition_observation_benchmark`、`test_config_cognition_defaults`、`test_memory_manager`、`test_trajectory_metadata`、`test_run_agent/test_cognition_observation_benchmark`、`test_cognition_trace_report_cli` 依賴 Task 3–8，**此時失敗屬預期**。

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(cognition): port cognitive stack modules verbatim onto v0.20.0

Cat 1 of the v0.20 migration (docs/plans/2026-08-10-hermes-upgrade-v0.20-實作計畫.md):
14 agent/ modules + CLI script + 19 test files + plan docs, copied
unchanged from feat/cognitive-stack-v0.16 (c70037042)."
```

## Task 3: Cat 2 乾淨五檔 3-way（hook 1–4 隨之自動帶過）

**Files:** Modify: `agent/agent_init.py`、`agent/memory_provider.py`、`agent/trajectory.py`、`run_agent.py`、`tests/agent/test_memory_provider.py`

**Interfaces:**
- Produces: `agent._cognition_config` / `agent._current_cognitive_route` / `agent._current_turn_cognition_metadata`（agent_init）、`AIAgent._resolve_current_cognitive_route(original_user_message=..., messages=...)` 與 `AIAgent._default_consistency_verifier(prompt)`（run_agent）、`_save_trajectory_to_file(..., metadata=...)`（trajectory）。Task 7 的 hook 5 呼叫 `_resolve_current_cognitive_route`；Task 8 的 hook 7 呼叫 `_default_consistency_verifier`。

- [ ] **Step 1: 逐檔 3-way**

```bash
for f in agent/agent_init.py agent/memory_provider.py agent/trajectory.py \
         run_agent.py tests/agent/test_memory_provider.py; do
  git diff v2026.6.5 c70037042 -- "$f" | git apply -3 && echo "OK   $f" || echo "FAIL $f"
done
```

預期 5 個全 `OK`（2026-08-10 merge-tree 已驗證）。若 FAIL：留下衝突標記，依「保留 upstream 結構、塞入我們的區塊」手動解。

- [ ] **Step 2: 驗證 hook 2/3/4 落地**

```bash
grep -n "_cognition_config = _get_cognition_config" agent/agent_init.py   # 預期 1 處（~1837 附近）
grep -n "_resolve_current_cognitive_route\|_default_consistency_verifier" run_agent.py | head -5   # 預期方法定義存在
grep -n "cognition_trace" run_agent.py | head -3   # 預期 _save_trajectory 內 metadata 邏輯存在
grep -n "def prefetch_layered" agent/memory_provider.py   # 預期存在（PR2 default 實作）
```

- [ ] **Step 3: import smoke + 相關測試**

```bash
uv run python -c "import run_agent, agent.agent_init, agent.memory_provider, agent.trajectory; print('OK')"
uv run pytest tests/agent/test_memory_provider.py tests/agent/test_trajectory_metadata.py -q 2>&1 | tail -3
```

預期全綠。

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(cognition): re-apply clean 3-way deltas onto v0.20.0 (hooks 1-4 carried)

agent_init cognition config init, run_agent route-resolve/verifier methods
and trajectory metadata passthrough, memory_provider prefetch_layered —
all applied automatically per the 2026-08-10 merge-tree verification."
```

## Task 4: memory_manager / hermes_logging / test_hermes_logging 衝突手解

**Files:** Modify: `agent/memory_manager.py`（2 hunks）、`hermes_logging.py`（1 hunk）、`tests/test_hermes_logging.py`（1 hunk）

**Interfaces:**
- Produces: `MemoryManager.prefetch_ranked_for_policy(query, *, layers)`、`MemoryManager.last_policy_recall_metadata`、`MemoryManager.last_plasticity_metadata` — Task 7 的 hook 6 消費。

- [ ] **Step 1: 3-way 套用（預期出現衝突標記）**

```bash
for f in agent/memory_manager.py hermes_logging.py tests/test_hermes_logging.py; do
  git diff v2026.6.5 c70037042 -- "$f" | git apply -3; echo "$f exit=$?"
done
grep -c "^<<<<<<<" agent/memory_manager.py hermes_logging.py tests/test_hermes_logging.py
```

預期衝突數：memory_manager 2、hermes_logging 1、test_hermes_logging 1。

- [ ] **Step 2: 解 `memory_manager.py` hunk 1（`__init__` 內，雙方各自附加）**

保留 **upstream 全部新行**（`self._external_prefetch_timeout = ...` 起、經 sync-executor/futures 機制、至 `self._shutdown_drain_state = {...}` 止），接著補上**我們的兩個屬性**（放在 upstream 新行之後）：

```python
        self.last_policy_recall_metadata: dict[str, Any] = {
            "enabled": False,
            "count": 0,
            "policy_ids": [],
            "citations": [],
            "categories": [],
        }
        self.last_plasticity_metadata: dict[str, Any] = build_plasticity_metadata([])
```

（`build_plasticity_metadata` 的 import 在檔頭，已隨 3-way 自動帶過。）刪除三個衝突標記行。

- [ ] **Step 3: 解 `memory_manager.py` hunk 2（方法區，雙方各自附加）**

保留 **upstream 的方法**（`prefetch_all` 新簽名、`queue_prefetch_all` 等）於原位，接著保留**我們的方法**（`prefetch_for_policy`、`prefetch_ranked_for_policy` 等整組，衝突標記內 ours 側全部內容）。刪除衝突標記。

- [ ] **Step 4: 解 `hermes_logging.py`（import 區）**

ours 側為空、theirs 側是 `import queue` → 保留 theirs 的 `import queue`，刪除標記。我們的 `_stderr_noise_filter_installed` guard（`f11cfddef`）在衝突區外，已自動合併（下一步驗證）。

- [ ] **Step 5: 解 `tests/test_hermes_logging.py`（fixture，雙方各自附加）**

兩邊都要，合併為（ours 兩行在前、theirs 註解與 reset 在後）：

```python
    hermes_logging._stderr_noise_filter_installed = False
    monkeypatch.setenv("HERMES_DISABLE_STDERR_NOISE_FILTER", "1")
    # File handlers now live behind the async QueueListener, not on the root
    # logger; tear down any leaked from other xdist tests in this worker.
    hermes_logging._reset_queued_handlers()
```

- [ ] **Step 6: 驗證 + 測試**

```bash
grep -c "^<<<<<<<\|^=======\|^>>>>>>>" agent/memory_manager.py hermes_logging.py tests/test_hermes_logging.py
# 預期全 0
grep -n "_stderr_noise_filter_installed" hermes_logging.py | head -3   # 預期 guard 存活
uv run python -c "import agent.memory_manager, hermes_logging; print('OK')"
uv run pytest tests/agent/test_memory_manager.py tests/test_hermes_logging.py -q 2>&1 | tail -3
```

預期全綠。若 `test_memory_manager` 因 upstream 新簽名（如 `prefetch_all(query, *, session_id="")`）失敗，修**測試**對齊新簽名；我們的方法保持既有簽名。

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(cognition): merge memory_manager + logging deltas onto v0.20 (keep-both)

memory_manager: upstream sync-executor machinery + our policy/plasticity
recall metadata and layer-aware prefetch methods coexist.
hermes_logging: upstream queue import + our pytest stderr-filter guard."
```

## Task 5: cognition config block → `hermes_cli/config_defaults.py`

**Files:**
- Modify: `hermes_cli/config_defaults.py`（v2026.8.3 原版 ~831 行 `"auxiliary": {` 之前插入）
- Modify: `hermes_cli/config.py`（僅 `_FALLBACK_COMMENT` 尾端加註解；DEFAULT_CONFIG 部分**用 upstream，不套我們的 diff**）

**Interfaces:**
- Produces: `DEFAULT_CONFIG["cognition"]`（頂層 key）— `agent.cognition_config.get_cognition_config` 與 `tests/hermes_cli/test_config_cognition_defaults.py` 消費（後者 `from hermes_cli.config import DEFAULT_CONFIG`，re-export 保證相容）。

- [ ] **Step 1: 插入 cognition block**

在 `hermes_cli/config_defaults.py` 的 `"bedrock"` block 結束後、`    "auxiliary": {` 行之前，插入（與舊 config.py 1191–1213 行逐字相同）：

```python
    # Per-turn cognitive routing scaffold (PR1).
    # Disabled by default; when enabled, classifies each turn into
    # fast/standard/deep and surfaces routing metadata for downstream
    # layers (verification ladder, layered retrieval, consistency guard).
    "cognition": {
        "enabled": False,
        "fast_mode": {
            "max_chars": 160,
            "max_words": 28,
            "allow_urls": False,
            "allow_code_blocks": False,
        },
        "deep_mode_triggers": {
            "historical_questions": True,
            "code_changes": True,
            "risky_external_actions": True,
            "architecture_decisions": True,
        },
        "consistency_guard": {
            "enabled": True,
            "deep_mode_only": True,
        },
    },
```

- [ ] **Step 2: `_FALLBACK_COMMENT` 尾端加 YAML 註解**

`hermes_cli/config.py` 的 `_FALLBACK_COMMENT = """`（~3448 行）字串**結尾 `"""` 之前**附加：

```
#
# ── Cognitive Routing ──────────────────────────────────────────────────
# Per-turn fast/standard/deep classification scaffold. Disabled by
# default; when enabled, classifies each turn and surfaces routing
# metadata for downstream layers (verification, retrieval, etc).
#
# cognition:
#   enabled: true
#   fast_mode:
#     max_chars: 160
#     max_words: 28
#   deep_mode_triggers:
#     historical_questions: true
#     code_changes: true
#     risky_external_actions: true
#     architecture_decisions: true
#   consistency_guard:
#     enabled: true
#     deep_mode_only: true
```

- [ ] **Step 3: 驗證頂層位置（上次的 fuzzy-context 陷阱）**

```bash
uv run python -c "
from hermes_cli.config import DEFAULT_CONFIG
assert 'cognition' in DEFAULT_CONFIG, 'cognition 不在頂層！'
assert 'cognition' not in (DEFAULT_CONFIG.get('providers') or {}), 'cognition 被塞進 providers！'
assert 'cognition' not in (DEFAULT_CONFIG.get('bedrock') or {}), 'cognition 被塞進 bedrock！'
print('config cognition block OK:', sorted(DEFAULT_CONFIG['cognition'].keys()))"
```

預期印出 `['consistency_guard', 'deep_mode_triggers', 'enabled', 'fast_mode']`。

- [ ] **Step 4: config 測試全套**

```bash
uv run pytest tests/hermes_cli/test_config_cognition_defaults.py -q 2>&1 | tail -2
uv run pytest tests/hermes_cli -q 2>&1 | tail -3
```

第一行預期全綠；第二行失敗集合須 ⊆ 乾淨基底失敗集合（Task 10 Step 4 會正式比對，此處先記錄數字）。

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(cognition): move cognition defaults into hermes_cli/config_defaults.py

Upstream extracted DEFAULT_CONFIG out of config.py; the cognition block
now lives in config_defaults.py top level (re-exported via config.py),
and the YAML usage comment re-lands at the _FALLBACK_COMMENT tail."
```

## Task 6: Hook 5/6 → `agent/turn_context.py`；conversation_loop 用 upstream 原版

**Files:**
- Modify: `agent/turn_context.py`（2 處，錨點見下）
- `agent/conversation_loop.py`：**不動**（分支開自 v2026.8.3，保持 upstream 原版即可；確認即可）

**Interfaces:**
- Consumes: `agent._resolve_current_cognitive_route`（Task 3）、`agent.retrieval_policy.resolve_retrieval_policy`（Task 2）、`MemoryManager.prefetch_ranked_for_policy` + metadata 屬性（Task 4）。
- Produces: `TurnContext.ext_prefetch_cache` 內容改由 layer-aware 路徑產生（cognition off 時 bit-for-bit 等同 upstream）。

- [ ] **Step 1: Hook 5 — route resolve**

錨點：`agent/turn_context.py` 內 `# Notify memory providers of the new turn (BEFORE prefetch_all).`（v2026.8.3 ~1147 行）。其**前**插入：

```python
    # ── Cognitive routing (PR1) ─────────────────────────────────────────
    # Resolve route metadata before any prefetch / model selection so
    # downstream layers can consume it. Disabled by default; when off,
    # this stays a no-op and behavior matches upstream exactly.
    agent._resolve_current_cognitive_route(
        original_user_message=original_user_message,
        messages=messages,
    )

```

- [ ] **Step 2: Hook 6 — layer-aware prefetch**

把 upstream 的 prefetch 區塊（`# External memory provider: prefetch once before the tool loop.` 註解起，到其 `except Exception:` / `pass` 止，~1155–1166 行）**整塊換成**：

```python
    # External memory provider: prefetch once before the tool loop.
    #
    # Skip prefetch on trivial prompts (greetings, acknowledgements) to
    # prevent memory-context injection on turns that carry no semantic signal.
    #
    # PR2: when the cognitive router has produced a route for this turn
    # the prefetch goes through the layer-aware orchestration so the
    # provider only consults the requested memory layers (principles /
    # semantic / episodic). When no route exists (cognition disabled or
    # router returned None) fall back to the legacy prefetch_all path so
    # behavior is bit-for-bit identical to upstream.
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            if not is_trivial_prompt(_query):
                from agent.retrieval_policy import resolve_retrieval_policy

                _policy = resolve_retrieval_policy(agent._current_cognitive_route)
                if _policy is not None:
                    ext_prefetch_cache = agent._memory_manager.prefetch_ranked_for_policy(
                        _query, layers=_policy.layers
                    ) or ""
                    try:
                        _policy_meta = getattr(
                            agent._memory_manager, "last_policy_recall_metadata", None
                        )
                        if isinstance(_policy_meta, dict) and isinstance(
                            agent._current_turn_cognition_metadata, dict
                        ):
                            agent._current_turn_cognition_metadata.update({
                                "policy_memory_enabled": bool(_policy_meta.get("enabled")),
                                "policy_memory_count": int(_policy_meta.get("count") or 0),
                                "policy_memory_ids": list(_policy_meta.get("policy_ids") or []),
                                "policy_memory_citations": list(_policy_meta.get("citations") or []),
                                "policy_memory_categories": list(_policy_meta.get("categories") or []),
                            })
                    except Exception:
                        pass
                    try:
                        _plasticity_meta = getattr(
                            agent._memory_manager, "last_plasticity_metadata", None
                        )
                        if isinstance(_plasticity_meta, dict) and isinstance(
                            agent._current_turn_cognition_metadata, dict
                        ):
                            agent._current_turn_cognition_metadata.update(_plasticity_meta)
                    except Exception:
                        pass
                else:
                    ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass
```

改寫重點（與 v0.16 版的差異）：變數名 `_ext_prefetch_cache` → **`ext_prefetch_cache`**（對齊 upstream，`TurnContext` 建構時取用）；**`is_trivial_prompt` 閘門包住兩條路徑**（cognition on 時瑣碎輸入同樣不 prefetch；cognition off 時行為與 upstream 逐位元相同）。`is_trivial_prompt` 該檔 44 行已 import，不用加。

- [ ] **Step 3: 語法 + import smoke + 確認 conversation_loop 乾淨**

```bash
uv run python -m py_compile agent/turn_context.py
uv run python -c "import agent.turn_context, agent.conversation_loop, run_agent; print('hooks 5-6 OK')"
git diff v2026.8.3 -- agent/conversation_loop.py | wc -l   # 預期 0（upstream 原版）
```

- [ ] **Step 4: 順序驗證（route resolve → on_turn_start → prefetch）**

```bash
uv run python -c "
import inspect
from agent.turn_context import build_turn_context
src = inspect.getsource(build_turn_context)
i_route = src.index('_resolve_current_cognitive_route(')
i_turn = src.index('.on_turn_start(')
i_pf = src.index('prefetch_ranked_for_policy(')
assert i_route < i_turn < i_pf, (i_route, i_turn, i_pf)
print('ordering OK')"
```

- [ ] **Step 5: Commit**

```bash
git add agent/turn_context.py
git commit -m "feat(cognition): re-wire hooks 5-6 into v0.20 turn_context

Upstream extracted the turn prologue into build_turn_context(); per-turn
route resolve and PR2 layer-aware prefetch move there. conversation_loop
stays pure upstream. is_trivial_prompt gate now guards both prefetch paths."
```

## Task 7: Hook 7/8 + cognition imports → `agent/turn_finalizer.py`

**Files:** Modify: `agent/turn_finalizer.py`（3 處：import 區、`finalize_turn` 中段、result dict）

**Interfaces:**
- Consumes: Task 2 模組函式（consistency_guard / process_monitor / autonomy_telemetry / cognition_trace）、`agent._default_consistency_verifier`（Task 3）、`agent._current_turn_cognition_metadata`。
- Produces: result dict 的 `cognition_metadata` / `cognition_trace` 欄位（gateway 與測試 bank 消費）；`agent._current_turn_cognition_metadata["cognition_trace"]`（hook 3 的 `_save_trajectory` 消費——呼叫點在本函式內、pipeline 之後，順序正確）。

- [ ] **Step 1: 加 imports**

`agent/turn_finalizer.py` 檔頭 `from agent.message_content import flatten_message_text` 之後加：

```python
from agent.autonomy_telemetry import (
    build_autonomy_metadata,
    build_autonomy_telemetry_from_metadata,
)
from agent.cognition_trace import build_cognition_turn_trace
from agent.consistency_guard import (
    resolve_verification_ladder,
    resolve_verification_plan,
    run_full_consistency_check,
    run_light_consistency_check,
    should_run_consistency_guard,
)
from agent.process_monitor import (
    assess_claims,
    build_process_monitor_metadata,
    extract_claims_from_response,
)
```

- [ ] **Step 2: Hook 7 — 擷取 post-generation pipeline 並插入**

```bash
git show c70037042:agent/conversation_loop.py | sed -n '4657,4814p' > /tmp/hook7_pipeline.py
head -1 /tmp/hook7_pipeline.py  # 預期：    # ── Consistency guard / verification second pass (PR3) ───────────
tail -1 /tmp/hook7_pipeline.py  # 預期：            _cognition_trace = None
grep -c "self\." /tmp/hook7_pipeline.py  # 預期 0（已是 agent. 形態）
```

插入位置：`finalize_turn()` 內 `# Post-loop cleanup must never lose the response.` 註解（v2026.8.3 ~233 行，preflight-rollback 區塊之後）**之前**。縮排同為 4 空格層級，逐字貼入，前後各留一空行。

注意：pipeline 內的 `logger` 由 `finalize_turn` 開頭的 `from agent.conversation_loop import logger` 提供（upstream 既有 lazy import），logger 名維持 `agent.conversation_loop`——測試 caplog 斷言不受影響。

- [ ] **Step 3: Hook 8 — result dict 加欄位**

錨點：`turn_finalizer.py` result dict 的 `"cost_source": agent.session_cost_source,`（~658 行，注意同檔另有 token-明細行，認準 result dict 組裝處）。其後插入：

```python
        "cognition_metadata": dict(agent._current_turn_cognition_metadata)
        if isinstance(agent._current_turn_cognition_metadata, dict)
        else {},
        "cognition_trace": _cognition_trace,
```

（`_cognition_trace` 由 Step 2 的 pipeline 定義，同函式作用域，已驗證無名稱衝突。）

- [ ] **Step 4: 語法 + import smoke + 順序驗證**

```bash
uv run python -m py_compile agent/turn_finalizer.py
uv run python -c "import agent.turn_finalizer, agent.conversation_loop, run_agent; print('hooks 7-8 OK')"
uv run python -c "
import inspect
from agent.turn_finalizer import finalize_turn
src = inspect.getsource(finalize_turn)
i_guard = src.index('Consistency guard / verification second pass')
i_save = src.index('._save_trajectory(')
i_dict = src.index('\"cognition_trace\": _cognition_trace')
assert i_guard < i_save < i_dict, (i_guard, i_save, i_dict)
print('finalizer ordering OK')"
```

- [ ] **Step 5: Commit**

```bash
git add agent/turn_finalizer.py
git commit -m "feat(cognition): re-wire hooks 7-8 into v0.20 turn_finalizer

Upstream extracted the post-loop tail into finalize_turn(); the PR3/PR16/
PR18/PR7 post-generation pipeline and the result-dict cognition fields
move there verbatim (already agent.-form). Pipeline runs before
_save_trajectory so trajectory metadata still sees cognition_trace."
```

## Task 8: `tests/run_agent/test_run_agent.py` 測試 bank 搬移與重指向

**Files:** Modify: `tests/run_agent/test_run_agent.py`

- [ ] **Step 1: 3-way 套用（1 個衝突，theirs 側為空）**

```bash
git diff v2026.6.5 c70037042 -- tests/run_agent/test_run_agent.py | git apply -3; echo "exit=$?"
grep -c "^<<<<<<<" tests/run_agent/test_run_agent.py   # 預期 1
```

解法：該 hunk theirs 側為空（純 context 相鄰），**保留 ours 側全部內容**（`TestMemoryProviderTurnStart` class 起），刪除三個標記行。

- [ ] **Step 2: 重指向 source-level 檢查（run_conversation → 新模組）**

先定位（合併後行號會位移，用 grep 找）：

```bash
grep -n "from agent.conversation_loop import run_conversation" tests/run_agent/test_run_agent.py
```

我們的測試 bank 有多處 `inspect.getsource(run_conversation)` 檢查（舊檔 ~6265–6376 行區域，約 5 處）。逐一改為新歸屬：

- `on_turn_start` 先於 `prefetch` 的檢查、`on_turn_start(agent._user_turn_count` 拼寫檢查、PR2 prefetch 存在性檢查、route-resolve 呼叫檢查 → 改 import `from agent.turn_context import build_turn_context` 並 `inspect.getsource(build_turn_context)`。注意 prefetch 順序檢查中 `.prefetch_all(` 的 index 應改為先找 `.prefetch_ranked_for_policy(`（layer-aware 路徑在前）或保留 `.prefetch_all(`（fallback 在後）——依各測試斷言語意逐一對應，斷言意圖不變。
- 若有針對 guard/pipeline 的 source 檢查 → 改 `from agent.turn_finalizer import finalize_turn` + `inspect.getsource(finalize_turn)`。

- [ ] **Step 3: 重指向 mock path（conversation_loop → turn_finalizer）**

```bash
sed -i '' 's/patch("agent\.conversation_loop\.run_light_consistency_check/patch("agent.turn_finalizer.run_light_consistency_check/g;
s/patch("agent\.conversation_loop\.run_full_consistency_check/patch("agent.turn_finalizer.run_full_consistency_check/g;
s/patch("agent\.conversation_loop\.extract_claims_from_response/patch("agent.turn_finalizer.extract_claims_from_response/g;
s/patch("agent\.conversation_loop\.build_autonomy_telemetry_from_metadata/patch("agent.turn_finalizer.build_autonomy_telemetry_from_metadata/g' \
  tests/run_agent/test_run_agent.py
grep -c 'patch("agent\.conversation_loop\.\(run_\|extract_\|build_autonomy\)' tests/run_agent/test_run_agent.py  # 預期 0
```

`caplog.at_level(..., logger="agent.conversation_loop")` **不改**（logger 名經 lazy import 保留）。

- [ ] **Step 4: 跑 cognitive 相關測試並逐一修綠**

```bash
uv run pytest tests/run_agent/test_run_agent.py -q -k "cogniti or route or verification or consistency or turn_start or prefetch" 2>&1 | tail -5
```

預期全綠。失敗逐一修（fixture / import path / 新簽名），**不改產品碼**。已知可能差異：AIAgent fixture 需要的參數在 v0.20 或有增減（比照 v0.16 遷移時 `base_url` 的處理）。

- [ ] **Step 5: 跑整檔**

```bash
uv run pytest tests/run_agent/test_run_agent.py -q 2>&1 | tail -3
```

判定：失敗集合 ⊆ 乾淨 v2026.8.3 基底同檔失敗集合（如需基底數據，`git stash && git switch --detach v2026.8.3 && uv run pytest tests/run_agent/test_run_agent.py -q 2>&1 | tail -3 && git switch feat/cognitive-stack-v0.20 && git stash pop`）。

- [ ] **Step 6: Commit**

```bash
git add tests/run_agent/test_run_agent.py
git commit -m "test(cognition): port cognitive test bank to v0.20 module layout

Source-level checks repointed to turn_context.build_turn_context /
turn_finalizer.finalize_turn; consistency/telemetry mock paths moved
agent.conversation_loop.* -> agent.turn_finalizer.*."
```

## Task 9: `test_cognition_observation_benchmark`（run_agent 側）與 scripts 測試收攏

**Files:** 視測試結果 Modify: `tests/run_agent/test_cognition_observation_benchmark.py`、`tests/scripts/test_cognition_trace_report_cli.py`

- [ ] **Step 1: 跑剩餘兩個依賴 hook 的測試檔**

```bash
uv run pytest tests/run_agent/test_cognition_observation_benchmark.py \
  tests/scripts/test_cognition_trace_report_cli.py \
  tests/agent/test_cognition_observation_benchmark.py -q 2>&1 | tail -3
```

- [ ] **Step 2: 失敗逐一修**

預期失敗模式與 Task 8 相同（模組搬家、AIAgent fixture 簽名）。對 `run_agent` 側 benchmark fixture，比照 v0.16 遷移 commit `ab3062713` 的手法（fixture 對齊 agent_init 的必要參數）。修到全綠。

- [ ] **Step 3: Commit（若有變更）**

```bash
git add -A && git commit -m "test(cognition): fix observation-benchmark fixtures for v0.20 module layout"
```

## Task 10: 全面驗證

- [ ] **Step 1: 完整 import smoke**

```bash
uv run python -c "
import run_agent, agent.conversation_loop, agent.agent_init, agent.turn_context, \
       agent.turn_finalizer, agent.codex_runtime
import agent.cognitive_router, agent.cognition_config, agent.cognition_trace, \
       agent.cognition_trace_report, agent.cognition_observation_benchmark, \
       agent.consistency_guard, agent.uncertainty_policy, agent.retrieval_policy, \
       agent.process_monitor, agent.autonomy_telemetry, agent.memory_ranker, \
       agent.memory_plasticity, agent.policy_memory, agent.interaction_stance
print('full import smoke OK')"
```

- [ ] **Step 2: 全部 cognitive 測試（對照 Task 0 Step 2 基準）**

```bash
uv run pytest tests/agent/test_cognitive_router.py tests/agent/test_cognition_config.py \
  tests/agent/test_consistency_guard.py tests/agent/test_uncertainty_policy.py \
  tests/agent/test_retrieval_policy.py tests/agent/test_cognition_trace.py \
  tests/agent/test_cognition_trace_report.py tests/agent/test_cognition_observation_benchmark.py \
  tests/agent/test_process_monitor.py tests/agent/test_autonomy_telemetry.py \
  tests/agent/test_memory_ranker.py tests/agent/test_interaction_stance.py \
  tests/agent/test_memory_manager.py tests/agent/test_memory_plasticity.py \
  tests/agent/test_policy_memory.py tests/agent/test_trajectory_metadata.py \
  tests/hermes_cli/test_config_cognition_defaults.py \
  tests/run_agent/test_cognition_observation_benchmark.py \
  tests/scripts/test_cognition_trace_report_cli.py -q 2>&1 | tail -3
```

預期：0 failed，passed 數 ≥ Task 0 基準。

- [ ] **Step 3: 我們動過的 upstream 測試檔**

```bash
uv run pytest tests/run_agent/test_run_agent.py tests/test_hermes_logging.py \
  tests/agent/test_memory_provider.py -q 2>&1 | tail -3
```

預期 0 failed（或 ⊆ 基底失敗集合）。

- [ ] **Step 4: 廣域回歸（與乾淨基底比對）**

```bash
uv run pytest tests/agent tests/hermes_cli tests/run_agent -q -p no:cacheprovider --ignore=tests/agent/lsp 2>&1 | tail -5
```

判定標準：**失敗集合 ⊆ 乾淨 v2026.8.3 基底的失敗集合**。基底取法：

```bash
git stash && git switch --detach v2026.8.3
uv run pytest tests/agent tests/hermes_cli tests/run_agent -q --ignore=tests/agent/lsp 2>&1 | tail -3
git switch feat/cognitive-stack-v0.20 && git stash pop || true
```

- [ ] **Step 5: 本機 e2e**

```bash
uv run hermes chat -q "say hi in 3 words" 2>&1 | tail -3
```

預期數十秒內回覆；`agent.log` 無 cognition 相關 exception。

- [ ] **Step 6: 更新 `.upgrade-progress.md`（各任務結果 + 測試數字）並 commit**

```bash
git add .upgrade-progress.md && git commit -m "docs(upgrade): v0.20 migration verification results"
```

## Task 11: Push 與 runtime handoff

> ⚠️ 動 production（4 個 launchd gateway）。**Step 2 之前向無名確認時間窗。**

- [ ] **Step 1: Push 分支**

```bash
git push -u origin feat/cognitive-stack-v0.20
```

- [ ] **Step 2: runtime 切換**（工作樹 `/Users/wuming/.hermes/hermes-agent`，remote 名 `wuming` 指向 fork）

```bash
cd /Users/wuming/.hermes/hermes-agent
git status --porcelain        # 只有 trajectory_samples.jsonl（untracked）屬預期；其他變更先 stash 並記錄
git fetch wuming
git switch -c cognitive-stack-v0.20 wuming/feat/cognitive-stack-v0.20
uv sync 2>&1 | tail -3
```

- [ ] **Step 3: 確認 stale timeout config 仍在**

```bash
grep -A3 "openai-codex" ~/.hermes/config.yaml | grep stale_timeout_seconds   # 預期：1800（上次 handoff 已設）
```

- [ ] **Step 4: 重啟 4 個 gateway 並驗證**

```bash
launchctl list | grep -i hermes    # 查 label：main / tianji / tianquan / yuheng
launchctl kickstart -k gui/$(id -u)/<main-label>
launchctl kickstart -k gui/$(id -u)/<tianji-label>
launchctl kickstart -k gui/$(id -u)/<tianquan-label>
launchctl kickstart -k gui/$(id -u)/<yuheng-label>
```

驗證：telegram / slack / feishu 均 `connected` + e2e：

```bash
hermes chat -q "say hi in 3 words"
```

- [ ] **Step 5: 觀察期（24h）**：盯 `agent.log` 的 (a) `stale` kill、(b) cognition exception、(c) **upstream verification-gate 與 PR3 guard 的互動**（若同一回合同時觸發二次驗證造成回覆延遲異常，記錄案例，評估把 `cognition.consistency_guard.deep_mode_only` 保持 true 或關閉其一）。

- [ ] **Step 6: 收尾記錄**：`.upgrade-progress.md` 補 handoff 結果（pid、平台連線、e2e 輸出），commit + push。

## Task 12（後續，觀察期穩定後）: main 收尾

需要無名在 GitHub 操作分支保護：

- [ ] `git tag archive/cognitive-stack-v016-lineage origin/main && git push origin archive/cognitive-stack-v016-lineage`
- [ ] 無名解除 main 的 force-push 保護
- [ ] `git push --force origin feat/cognitive-stack-v0.20:main`
- [ ] 重新上保護；main 回歸「指向現行 production」語意

---

## 回滾程序

```bash
# repo 側：分支丟掉即可
git switch feat/cognitive-stack-v0.16

# runtime 側（若已 handoff）：
cd /Users/wuming/.hermes/hermes-agent
git switch cognitive-stack-v0.16
uv sync
# 重啟 4 個 gateway（同 Task 11 Step 4）
```

## 遺留與後續事項（維持 deferred，本次不碰）

1. **SSE-liveness stale detector 重移植**：config 緩解已在 runtime；若 v0.20 deep-mode 誤殺重現再啟動。
2. **smart-routing-plugin follow-up**：patches 在 `docs/plans/smart-routing-plugin-followup/`。
3. **Gemini Imagen**（`tools/image_generation_tool.py` 擴充）。
4. **新觀察項**：upstream verification-gate（`_pending_verification_response`）與 PR3 consistency guard 的長期關係——若二者語意重疊度高，評估下一輪把 PR3 guard 改掛在 upstream gate 之上或退役。
