# Hermes v0.21.5 正式切換與回復手冊

**狀態**：候選版已完成驗證；**以下步驟尚未執行，需無名確認時間窗後才可動正式環境。**
**日期**：2026-09-26
**紀錄**：`.upgrade-progress.md`「Hermes upgrade to v0.21.5」章節

## 版本

| 項目 | 值 |
|---|---|
| 官方基底 | tag `v2026.9.24` = v0.21.5，commit `f97608f178d1ffeca59860195ab7da295f7c8e5f`（tag object `e3dd27ee2d8b`） |
| 候選分支 | `feat/cognitive-stack-v0.21.5`（開發 repo；worktree `.worktrees/cognitive-stack-v0.21.5`） |
| 現行正式 | runtime `~/.hermes/hermes-agent` 分支 `cognitive-stack-v0.20` @ `e6910a9`，launchd 用 `venv/` |
| 還原標籤 | `pre-upgrade-20260926-1001/feat/cognitive-stack-v0.20` → `e6910a9`（本機，未推送） |

## 受影響程序

| 程序 | 目前 | 切換時 |
|---|---|---|
| `ai.hermes.gateway`（default） | PID 78139（原 4529；9/26 11:32 測試事件後由 launchd 重啟，仍是 v0.20），含 cron ticker、MCP 子程序 | 停止 → 新版啟動；首次開啟 `~/.hermes/state.db` 做 schema 25→30（副本實測約 48 秒） |
| `ai.hermes.gateway-tianji` / `-tianquan` / `-yuheng` | PID 4511 / 4516 / 4523 | 停止 → 新版啟動；各自 state.db 遷移 <1 秒 |
| 互動 CLI（`~/.local/bin/hermes` → 開發 repo `.venv`，v0.20） | PID 47631、77439 | 不會被切換；建議切換前先結束，避免遷移那一分鐘遇到 database is locked。v0.20 讀寫 v30 資料庫已在副本驗證可行 |

## 已知風險（切換前必讀）

1. **共用 Slack token 的搶鎖**（v0.20 就存在，v0.21.5 行為相同）：三個 profile 的 `.env` 都有同一組 Slack app token，plist 都帶 `--replace`。後啟動的 gateway 在啟動階段會終止持有 Slack 的其他 profile gateway；被終止者以 0 結束，launchd（`SuccessfulExit=false`）不會重開它。**逐一重啟會讓前面的 profile 掉線。** 本手冊用「三個 profile 同時載入」重現開機時的情況，但結果不保證；若有 profile 掉線，由無名決定 Slack 歸屬（需改該 profile 設定，本次未授權，不做）。
2. **主資料庫遷移**：11.47 GB，首次開啟阻塞約 1 分鐘，期間 default gateway 不回應。
3. **plist**：只用 `launchctl bootout/bootstrap`，不要用 `hermes gateway start/install/restart`（新版會依新模板重寫 plist）。
4. **Python runtime 自動修復**（v0.21.5 新功能）：目前的 Python（uv cpython 3.11.15，SQLite 3.50.4）會被判定有 WAL 重設漏洞。之後若執行 `hermes update`、`hermes setup` 的終端機後端設定、computer-use 或 browser-use CLI 安裝，新版會自動把 `~/.hermes/hermes-agent/venv` 換成新的受管理 Python，而且只裝 `--extra all`；telegram / slack / feishu 套件要等 gateway 下次啟動時按需補裝（需要網路）。本手冊維持 3.11.15（和現在相同）。若之後發生自動修復，停 gateway 後在 `~/.hermes/hermes-agent` 執行 `UV_PROJECT_ENVIRONMENT=venv uv sync --locked --extra all --extra messaging --extra slack --extra feishu --extra fal --extra edge-tts`（**不要加 `--python`**，保留修復後的新 Python），再啟動 gateway。

## 切換步驟

```bash
TS=$(date +%Y%m%d-%H%M)
RT=/Users/wuming/.hermes/hermes-agent
PY311=/Users/wuming/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11
BK=/Volumes/Data/hermes-backups/pre-v0.21.5-$TS
mkdir -p "$BK" && chmod 700 "$BK"

# 1) 停止 4 個 gateway（bootout = 停止且不被 KeepAlive 重開）；cron 在 default gateway 內，會一起暫停
for L in ai.hermes.gateway ai.hermes.gateway-tianji ai.hermes.gateway-tianquan ai.hermes.gateway-yuheng; do launchctl bootout gui/$(id -u)/$L; done
while pgrep -f 'hermes_cli.main.*gateway run' >/dev/null; do sleep 1; done   # 等 drain 結束
launchctl list | grep ai.hermes          # 預期：沒有輸出

# 2) 一致性備份（無寫入者；SQLite online backup）
"$PY311" - "$BK" <<'EOF'
import os, sqlite3, sys
bk, home = sys.argv[1], os.path.expanduser("~/.hermes")
dbs = {"state.db": f"{home}/state.db", "kanban.db": f"{home}/kanban.db",
       "cron-executions.db": f"{home}/cron/executions.db"}
for p in ("tianji", "tianquan", "yuheng"):
    dbs[f"{p}-state.db"] = f"{home}/profiles/{p}/state.db"
for name, src in dbs.items():
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(f"{bk}/{name}")
    s.backup(d, pages=-1); d.close(); s.close()
    print(name, "backed up")
EOF
# 備份檔完整性檢查不擋切換（11 GB 約需數分鐘），放背景跑，稍後看結果：
(for f in "$BK"/*.db; do echo "$(basename $f): $(sqlite3 "file:$f?mode=ro" 'PRAGMA quick_check;')"; done > "$BK/quick_check.txt" 2>&1 &)
cp -p ~/.hermes/config.yaml ~/.hermes/auth.json ~/.hermes/.env ~/.hermes/gateway_state.json ~/.hermes/channel_directory.json "$BK"/
cp -p ~/.hermes/cron/jobs.json "$BK"/cron-jobs.json
for p in tianji tianquan yuheng; do mkdir -p "$BK/profiles/$p"; cp -p ~/.hermes/profiles/$p/config.yaml ~/.hermes/profiles/$p/.env "$BK/profiles/$p/"; done
tar -C ~/.hermes -czf "$BK/skills.tgz" skills
git -C "$RT" rev-parse HEAD > "$BK/runtime-HEAD.txt"
uv pip freeze --python "$RT/venv/bin/python" > "$BK/runtime-venv-freeze.txt"

# 3) 程式：從本機開發 repo 取候選分支，runtime 切過去（未追蹤的 trajectory_samples.jsonl 不受影響）
git -C "$RT" fetch workspace feat/cognitive-stack-v0.21.5
git -C "$RT" switch -c cognitive-stack-v0.21.5 workspace/feat/cognitive-stack-v0.21.5
git -C "$RT" log --oneline -1            # 預期：候選分支最新 commit

# 4) 環境：舊 venv 保留原封不動，建立新 venv（extras = 現行正式組合 + 已懶載入的 fal / edge-tts）
mv "$RT/venv" "$RT/venv.v0.20-$TS"
(cd "$RT" && UV_PROJECT_ENVIRONMENT=venv uv sync --locked --python "$PY311" \
   --extra all --extra messaging --extra slack --extra feishu --extra fal --extra edge-tts)

# 5) import smoke（隔離 HERMES_HOME，不寫正式資料）
HERMES_HOME=$(mktemp -d) "$RT/venv/bin/python" -c "import run_agent, gateway.run, cron.scheduler, telegram, slack_bolt, aiohttp, agent.cognition_routing; from hermes_cli import __version__; print('ok', __version__)"

# 6) 啟動：default 先（做主資料庫遷移），確認上線後三個 profile 同時載入
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist
tail -f ~/.hermes/logs/gateway.log       # 看到 "telegram connected" 後 Ctrl-C
for p in tianji tianquan yuheng; do launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway-$p.plist & done; wait

# 7) 驗證
launchctl list | grep ai.hermes                                            # 4 個都有 PID
sqlite3 "file:$HOME/.hermes/state.db?mode=ro" "SELECT version FROM schema_version"   # 30
ls ~/.local/state/hermes/gateway-locks/                                    # 誰持有 slack / telegram
"$RT/venv/bin/hermes" --version                                            # v0.21.5
"$RT/venv/bin/hermes" chat -q "say hi in 3 words" -Q                      # 真實模型、正式資料
grep -iE 'cognit|consistency_guard|Traceback' ~/.hermes/logs/errors.log | tail
```

**選配（另需授權）**：CLI launcher 改用新版：`ln -sfn /Users/wuming/.hermes/hermes-agent/venv/bin/hermes /Users/wuming/.local/bin/hermes`。不改的話，互動 CLI 仍是 v0.20（可正常讀寫新資料庫）。

## 回復步驟

```bash
RT=/Users/wuming/.hermes/hermes-agent
OLD=$(ls -d "$RT"/venv.v0.20-* | tail -1)          # 切換時保留的舊 venv
echo "old venv: $OLD"                               # 必須有值才繼續
for L in ai.hermes.gateway ai.hermes.gateway-tianji ai.hermes.gateway-tianquan ai.hermes.gateway-yuheng; do launchctl bootout gui/$(id -u)/$L; done
while pgrep -f 'hermes_cli.main.*gateway run' >/dev/null; do sleep 1; done
git -C "$RT" switch cognitive-stack-v0.20                                 # e6910a9
mv "$RT/venv" "$RT/venv.v0.21.5-rollback-$(date +%Y%m%d-%H%M)" && mv "$OLD" "$RT/venv"
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist
for p in tianji tianquan yuheng; do launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway-$p.plist & done; wait
# 若切換時改過 launcher：
ln -sfn /Volumes/Data/openclaw/workspace/hermes-agent/.venv/bin/hermes /Users/wuming/.local/bin/hermes
```

**資料相容性**（全部在副本驗證）：

- state.db 維持 schema 30 即可：v0.20 在 v30 資料庫上讀取（對話、訊息、全文搜尋）與寫入（建立對話、新增訊息、標題、結束）都正常，新觸發器會自動補欄位。**回復程式不需要還原資料庫。**
- kanban.db、cron/executions.db：新版只加索引，v0.20 可照常開啟。
- config.yaml：新版不會在 gateway / CLI 啟動時自動遷移；462 個使用者設定值在兩版的實際生效值相同。
- auth.json：兩版格式版本都是 1。

**只有資料本身損壞時**才還原備份（會失去切換後的新對話）：停止所有 gateway 與 CLI → 將 `$BK/state.db` 複製回 `~/.hermes/state.db`，並先移除 `~/.hermes/state.db-wal`、`~/.hermes/state.db-shm`（profile 資料庫同理）→ 依上方步驟啟動。
