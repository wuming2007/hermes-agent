# Hermes 升級 v0.20.0 (2026.8.3) — cognitive stack 遷移設計規格

**日期**：2026-08-10
**狀態**：已核可（無名 2026-08-10）
**下一步**：依本 spec 產出實作計畫（superpowers:writing-plans）

## 目標

把 cognitive stack（PR1–18）從 `feat/cognitive-stack-v0.16`（基底 upstream `v2026.6.5` = v0.16.0）遷移到 upstream `v2026.8.3`（= **v0.20.0**），產出可上 production 的 `feat/cognitive-stack-v0.20` 分支，完成 runtime handoff，並於觀察期後把 main 更新到現行 lineage。

## 已確認的決策

| 決策 | 結論 |
|---|---|
| 目標版本 | **v0.20.0**（tag `v2026.8.3`）。使用者原提 v0.19，經確認後選最新版（v0.19.0=`v2026.7.20`、v0.19.1=`v2026.7.30` 不採用） |
| 範圍 | 完整：分支建立 → 遷移 → 驗證 → push → **runtime 切換**（重啟前確認時間窗） |
| 做法 | **方案 A：重建而非 rebase**（連續兩次升級驗證過的 playbook）。merge（衝突量大、看不到語意衝突）與 rebase（衝突品質差、無逐步驗證節奏）均否決 |
| main 收尾 | 觀察期（數日）後執行：archive tag 舊 lineage → 解除 GitHub 保護 → force-push `feat/cognitive-stack-v0.20` → main → 重新上保護 |

## 版本脈絡（2026-08-10 調查）

| 項目 | 值 |
|---|---|
| 目前分支 | `feat/cognitive-stack-v0.16`，tip = `7eab08218` |
| 目前基底 | `3c231eb39` = upstream v0.16.0（tag `v2026.6.5`） |
| 本地 delta | `git diff v2026.6.5..7eab08218`：9 commits、62 檔、+11,542 行 |
| 升級目標 | tag `v2026.8.3` = v0.20.0 |
| upstream 差距 | 9,860 commits / 6,865 檔（上次 v0.16 遷移為 2,231 commits，本次約 4 倍） |
| 整合點 churn | `conversation_loop.py` +3214/−1065、`run_agent.py` +2535/−240、`agent_init.py` +1133/−82、`hermes_cli/config.py` **+2412/−3408（大改寫）**、`memory_manager.py` +627/−39、`hermes_logging.py` +309/−9、`trajectory.py` **零變動** |
| 模組結構 | `agent/agent_init.py`、`agent/conversation_loop.py` 均仍存在；新增 `agent/agent_runtime_helpers.py` 等檔（需調查 AIAgent 方法是否搬移） |
| runtime 工作樹 | `~/.hermes/hermes-agent`，現在 `cognitive-stack-v0.16`，乾淨（僅 untracked `trajectory_samples.jsonl`） |

## 設計

### 1. 分支與安全網

- 從 `v2026.8.3` 開 `feat/cognitive-stack-v0.20`。
- 開工前打 `pre-upgrade-<TS>/feat/cognitive-stack-v0.16` 與 `pre-upgrade-<TS>/main` 安全 tag 並 push origin。
- `.upgrade-progress.md` 開新章節，逐 task 記錄（沿用既有格式）。

### 2. 檔案分類（四類法）

| 類別 | 檔案 | 處理方式 |
|---|---|---|
| **Cat 1**（我們新增，upstream 無） | `agent/` 14 個 cognitive 模組、`scripts/cognition_trace_report.py`、cognitive 測試檔（上次為 21 個；實作計畫時以 `git diff v2026.6.5..7eab08218 --name-only` 全量清單為準）、`docs/plans/` | `git checkout feat/cognitive-stack-v0.16 --` verbatim copy。計畫階段先全量確認 v0.20 未新增同名檔 |
| **Cat 2**（雙方都改） | `agent/memory_manager.py`、`agent/memory_provider.py`、`hermes_cli/config.py`、`hermes_logging.py`、`tests/agent/test_memory_provider.py`、`tests/cli/test_cli_provider_resolution.py`、`tests/gateway/test_fast_command.py`、`tests/test_hermes_logging.py` | `git diff v2026.6.5 7eab08218 -- <f> \| git apply -3`；失敗者手動合（保留 upstream 結構、塞入我們的區塊） |
| **Cat 3**（我們改、upstream 沒動） | `agent/trajectory.py`（`metadata=` 參數） | 3-way，必定乾淨（已驗證 upstream 零變動） |
| **Cat 4**（整合 hook 檔） | `run_agent.py`（hook 1/3/4）、`agent/agent_init.py`（hook 2）、`agent/conversation_loop.py`（hook 5/6/7/8）、`tests/run_agent/test_run_agent.py`（測試 bank） | **先試 3-way**（見 §3），衝突才降級手動重接 |

註：`hermes_logging.py` 的 delta 這次**多了 `f11cfddef`**（pytest 下不裝 stderr-noise-filter 的 guard）；已驗證 v0.20 upstream 無等效修正，必須帶過去。

### 3. Hook 重接策略

上次是「跨檔搬家」（run_agent.py 拆進 conversation_loop.py），8 個 hook 全手工。**本次結構優勢**：8 個 hook 已以 v0.16 形態（`agent.` 風格、模組層函式）住在對應檔案裡，upstream v0.20 未再拆檔。因此：

1. Cat 4 各檔先試 `git apply -3`。
2. 衝突的檔案（預期至少 `conversation_loop.py`、`run_agent.py`）降級為手動重接：錨點在 `v2026.8.3` 上**重新調查**（route resolve 呼叫點、prefetch 區塊、post-generation pipeline 插入點、result dict 欄位），不沿用任何舊行號。
3. 8-hook 清單與語意沿用上次 playbook（imports / init / trajectory metadata / 2 個 AIAgent 方法 / route resolve / layer-aware prefetch / post-generation pipeline / result dict）。
4. 計畫階段必查的語意衝突點：cognitive 模組外部依賴（`tools.registry.tool_error` 等）在 v0.20 是否存在；`memory_manager` 介面（`prefetch_all` / `prefetch_ranked_for_policy` 掛載點、`last_policy_recall_metadata` / `last_plasticity_metadata`）是否相容；AIAgent 方法是否搬進 `agent_runtime_helpers.py` 等新模組。

### 4. codex fixes 與 deferred 事項

- `providers.openai-codex.stale_timeout_seconds` config 緩解已在 runtime config（上次 handoff 設定），只需驗證 v0.20 讀取路徑仍存在。
- 維持 deferred、本次不碰：SSE-liveness stale detector 重移植、smart-routing-plugin follow-up、Gemini Imagen（`tools/image_generation_tool.py` 擴充）。

### 5. 驗證標準

- 每個 task 結束跑 import smoke（cognitive 模組全量 + `run_agent` + `agent.conversation_loop` + `agent.agent_init`）。
- Cognitive 測試套件全綠（基準 = 現分支 `7eab08218` 上的全綠集合；實作計畫時先在現分支跑一次記錄數字）。
- 我們動過的 upstream 測試檔（`test_run_agent.py`、`test_hermes_logging.py` 等 5 檔）0 failed。
- 廣域回歸判定：**失敗集合 ⊆ 乾淨 `v2026.8.3` 基底的失敗集合**（先在裸基底取基準）。
- `hermes_cli/config.py` 大改寫加驗：cognition block 在 DEFAULT_CONFIG 頂層（非 `providers` 內）+ config 相關測試全套。
- 本機 e2e：`uv run hermes chat -q "say hi in 3 words"`。

### 6. Runtime handoff、觀察期、main 收尾

1. Push `feat/cognitive-stack-v0.20`。
2. Runtime 工作樹：fetch → 切 `cognitive-stack-v0.20` → `uv sync`。
3. **重啟 4 個 launchd gateway（main / tianji / tianquan / yuheng）前向無名確認時間窗**。
4. 驗證：telegram / slack / feishu 均 `connected` + e2e chat。
5. 觀察期 24h：盯 agent.log 的 `stale` kill 與 cognition exception。
6. 穩定數日後 main 收尾：`archive/cognitive-stack-old-lineage` tag → 無名解除 main force-push 保護 → force-push → 重新上保護。

## 風險與緩解

| 風險 | 緩解 |
|---|---|
| `config.py` 大改寫，3-way 大概率失敗 | 準備手動安置 fallback；驗證 cognition block 頂層位置（上次「塞進 providers」陷阱） |
| `memory_manager.py` +627 行，介面可能變 | 計畫階段先 diff 檢查掛載點方法簽名；測試 `test_memory_manager.py` 守門 |
| 新模組拆分（`agent_runtime_helpers.py` 等） | 計畫階段調查 AIAgent 方法歸屬，hook 3/4 錨點隨之修正 |
| merge-tree / 3-way 看不到語意衝突 | 每 task import smoke + 測試守門；外部依賴逐一驗證（上次 `smart_model_routing` 教訓） |
| v0.20 依賴 pin / Python 版本上限變動 | Task 1 `uv sync` 即驗證；`requires-python` 範圍先查 |

## 回滾程序

- **repo 側**：`git switch feat/cognitive-stack-v0.16`（舊分支與安全 tag 原封不動）。
- **runtime 側**（若已 handoff）：`git switch cognitive-stack-v0.16` → `uv sync` → 重啟 4 個 gateway。

## 成功標準

1. `feat/cognitive-stack-v0.20` 上所有驗證標準（§5）通過。
2. Production 4 gateway 在 v0.20 上運行、平台連線正常、24h 無 cognition exception 與 stale 誤殺。
3. main 指向現行 production lineage（觀察期後）。
