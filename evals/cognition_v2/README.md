# Cognition v2 phase 1 evaluation

Measures what the working memory (`cognition_wm` plugin) and the evidence loop change with a
real model: the same cases run twice, OFF (today's behavior: cognition routing on, no working
memory, no evidence loop) and ON (both enabled), with the model configured in a dedicated eval
home.

The scripted, reproducible version of these scenarios is the test suite
`tests/e2e/cognition_v2/test_scenarios.py`; this harness is for the real model.

## Safety

- Runs only in an eval home created with `--init-home`, never in the everyday `~/.hermes`.
- The user logs in to the eval home themselves. Nothing copies credentials from another home.
- Toolsets are `file`, `terminal` and `cognition_wm`: no messaging, web, cron or subagents.
  Every case runs in its own directory under `<eval home>/eval-work/`.
- Turns run non-interactively, so an action that needs human approval is denied, exactly as in
  `hermes chat -q`.
- The eval home's `config.yaml` is restored after the run.

## Procedure

```bash
python evals/cognition_v2/runner.py --init-home ~/hermes-eval-home
HERMES_HOME=~/hermes-eval-home hermes login          # once, by the user
# pick the model in ~/hermes-eval-home/config.yaml (the one used every day)
python evals/cognition_v2/runner.py --home ~/hermes-eval-home \
    --cases evals/cognition_v2/cases.example.jsonl --out results.jsonl --set dev --repeats 3
python evals/cognition_v2/runner.py --summarize results.jsonl
```

- **Cases**: `cases.example.jsonl` holds de-identified development cases (S0–S6). Keep a
  holdout set of about ten more in a private file, marked `"set": "holdout"`. Do not read or tune
  on them during development; run them once at the end with `--set holdout`.
- **Arms and repeats**: OFF and ON, three repeats each, same model.
- **Per turn** (one JSONL row): wall time, API calls, prompt and completion tokens, estimated
  cost, tool calls, blocked tool calls, snapshot characters added to the user message,
  evidence-loop rounds, unsupported claims, and whether the answer was marked.
- **Scoring** (by reading the rows, with a human spot check):
  - resumed the task correctly after compaction or a new run (S1);
  - execution tool calls after a pause or cancel (S2) and in prompt-only requests (S3);
  - unsupported "done" claims that reach the user unmarked (S4, S6);
  - the current fact used after an update (S5);
  - OFF against ON cost per turn (S0 must show no difference in calls).
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
