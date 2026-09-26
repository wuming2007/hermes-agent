# Cognition v2 evaluation

Measures what cognition v2 changes with a real model: the same cases run in three arms with the
model configured in a dedicated eval home — `off` (the existing cognition: routing on, no working
memory, no evidence loop), `on` (phase 1: the working memory `cognition_wm` plugin and the
evidence loop) and `v2` (phase 1 plus learning and calibration).

The scripted, reproducible versions are `tests/e2e/cognition_v2/test_scenarios.py` (phase 1) and
`tests/e2e/cognition_v2/test_phase2_scenarios.py` (phase 2); this harness is for the real model.

## Safety

- Runs only in an eval home created with `--init-home`, never in the everyday `~/.hermes`.
- The user logs in to the eval home themselves. Nothing copies credentials from another home.
- `--init-home` sets `auth.adopt_external_logins: false`, so the eval home never borrows the Codex
  CLI or Claude Code login (their refresh tokens rotate, so borrowing can log those programs out).
  The runner refuses a home where it is on.
- Toolsets are `file`, `terminal` and `cognition_wm`: no messaging, web, cron or subagents.
  Every case runs in its own directory under `<eval home>/eval-work/`.
- A separate HERMES_HOME and work directory are not a sandbox, so the run is re-executed under a
  macOS Seatbelt profile (`sandbox.py`; it refuses to run where `sandbox-exec` is missing): the real
  home and `/Volumes/Data` can be neither read nor written except the eval home, this checkout and
  the Python install; the run gets its own HOME inside the eval home; `claude`, `codex` and the other
  coding agents on its PATH are stand-ins that only log the call (`fake-bin/calls.log`); running the
  real agents, `launchctl`, `osascript`, `open`, `crontab`, `sudo`, `security`, `hermes` and the usual
  network clients is denied.
- What stays open: network (the model's API) and reading the eval home, including its own login.
  Every call a case made is in its result row (`calls`) for review.
- Turns run non-interactively, so an action that needs human approval is denied, exactly as in
  `hermes chat -q`.
- The eval home's `config.yaml` is restored after the run.

## Procedure

```bash
python evals/cognition_v2/runner.py --init-home ~/hermes-eval-home
HERMES_HOME=~/hermes-eval-home hermes auth add openai-codex --type oauth   # once, by the user;
#   use the provider you use every day, and answer N if it offers to import another program's login
# pick the model in ~/hermes-eval-home/config.yaml (the one used every day)
python evals/cognition_v2/runner.py --home ~/hermes-eval-home \
    --cases evals/cognition_v2/cases.phase2.example.jsonl --out results.jsonl --set dev --repeats 3
python evals/cognition_v2/runner.py --summarize results.jsonl
```

- **Cases**: `cases.example.jsonl` (phase 1, S0–S6) and `cases.phase2.example.jsonl` (phase 2,
  L1–L9 and claim quality) hold de-identified development cases. Keep a holdout set of about ten
  more in a private file, marked `"set": "holdout"`. Do not read or tune on them during
  development; run them once at the end with `--set holdout`.
- **Arms and repeats**: `off`, `on` and `v2` (`--arms`), three repeats each, same model. Every
  case run starts with nothing learned or recorded, so runs and cases stay independent.
- **Per turn** (one JSONL row, `kind: turn`): wall time, API calls, prompt and completion tokens,
  estimated cost, tool calls, blocked tool calls, working-memory and lessons characters added to
  the user message, evidence-loop rounds, unsupported claims, whether the answer was marked,
  whether the answer check ran, the track-record band, and the case's checks for that turn.
- **Per case run** (`kind: case`): the lessons kept by status, and the recorded claims by executor
  and result (for example, whether a coding agent's report was counted as ours).
- **Scoring** (by reading the rows, with a human spot check):
  - resumed the task correctly after compaction or a new run (S1);
  - execution tool calls after a pause or cancel (S2) and in prompt-only requests (S3);
  - unsupported "done" claims that reach the user unmarked (S4, S6);
  - the current fact used after an update (S5);
  - OFF against ON cost per turn (S0 must show no difference in calls).
  - phase 2: the share of checks that held per arm (`--summarize`): a correction carried into a new
    conversation (L1), not carried into another project or other work (L2), a withdrawn lesson
    gone (L3), limits and stops held (L6); reports never counted as our success (L4, case rows);
    answer checks where the record is poor and nowhere else (L8, L7); the same check counted once
    (L9); unsupported claims, latency and tokens per arm.
- **Reporting**: medians and ranges per arm (`--summarize`). Set any improvement threshold only
  after the OFF baseline has been measured; never fill in numbers that were not run.

## Case format

One JSON object per line:

| Key | Meaning |
|---|---|
| `id` | case name |
| `set` | `dev` or `holdout` |
| `turns` | the user messages, one agent run each |
| `files` | files written into the case directory before the first turn |
| `files_before_turn` | `{"<turn index>": {path: content}}` written just before that turn |
| `compact_before_turn` | turn index before which the conversation is compacted (`/compress`) |
| `sessions` | instead of `turns`: conversations in order, each `{turns, project?, files?, files_before_turn?, compact_before_turn?}`; `project` is a subdirectory (another project) |
| `expect` | checks, each `{session, turn, <kind>: value}` with kind `ran`, `not_ran` (a shell command of that turn that ran, blocked ones do not), `answer_has`, `answer_lacks`, or `file` plus `file_has`, `file_lacks`, `file_exists` |
| `seed_outcomes` | earlier results for the eval model, `[{task_kind, claim_class, results}]`, recorded in the case directory before it starts |
