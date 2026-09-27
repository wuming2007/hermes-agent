---
title: Working memory and the evidence loop
description: Keep task state, the user's limits and cancellations across turns, and stop unverified "done" claims
---

# Working memory and the evidence loop

Two opt-in parts of cognitive routing (`cognition.enabled: true`) that keep the agent honest about
long-running work:

- **Working memory** (the bundled `cognition_wm` plugin) keeps each task's state — goal,
  deliverables, decisions, facts, the user's limits, pauses and cancellations, and what the user
  allowed — across turns, resumed chats and context compaction.
- **The evidence loop** checks an answer's state claims ("deployed", "tests pass", "fixed",
  "done") against the tool results of the turn before the answer is stored.

## Enable

```yaml
plugins:
  enabled: [cognition_wm]        # working memory
cognition:
  enabled: true
  evidence_loop:
    enabled: true                # evidence loop
```

Both take effect in the next session.

## What the model sees

At the end of the user's message (never in the system prompt, so the prompt cache prefix does
not change) the plugin adds a snapshot fenced as `<working-memory>`: the focus task with its
status, goal, deliverables, decisions, facts, limits and permissions, other open tasks of the
project, and evidence rules. When nothing changed since the last snapshot in the conversation,
only a one-line pointer is added. A conversation with no task adds nothing.

The model records state with the `task_state` tool. Only a turn that carries the user's own words
can create, pause or cancel a task or grant a permission; notices, subagent results and tool
output cannot.

## What it enforces

- A paused or cancelled task stays stopped: calls that act are blocked, also under `/yolo` —
  writes, repo-changing git, coding agents, deploys and restarts, migrations, messages, clicks
  and keystrokes in a browser or on the desktop, Home Assistant services, board work and paid
  generation. A call whose effect cannot be told (a script such as `python3 refactor.py`, page
  script, an unknown plugin tool, an MCP tool not declared read-only) needs your confirmation,
  and is blocked under `/yolo`. Reading stays open: pages, screenshots, status, searches, and
  listing or stopping jobs and subagents.
- A limit the user just gave ("only the prompt, don't change the repo") sends execution tools to
  the approval prompt until the model records it.
- A permission the user gave in an earlier session must be confirmed again. While the user forbids
  an action for the task, a command whose effect cannot be told goes to the approval prompt.
- Shell commands are read however they are wrapped: chains, pipes, several lines, `sudo`/`env`/
  `timeout` prefixes, `bash -c`, `eval`, `npx`, command substitution, and `python -c`/`node -e`
  one-liners. Tool calls made inside `execute_code` are gated like direct calls.
- Working-memory state is never written into long-term memory.
- An earlier session's record (`session_search`) says what was so then: a state claim that only
  such a record backs is marked unverified with the record's time, unless the answer cites it
  ("according to the last session", "as of …"). The snapshot dates the facts and finished steps it
  carries. The assistant's own records (task state, todos, memory) are never evidence.
- With the evidence loop on, an answer with unbacked claims gets one check round (at most
  `max_extra_tool_calls` calls: reads, test runs and commands that check something, never a write,
  repo change, deploy, migration or agent launch); claims that stay unbacked are marked as
  unverified in the answer that is stored and shown.

## Lessons and the track record

Two further opt-in parts, both off by default:

```yaml
plugins:
  entries:
    cognition_wm:
      settings:
        learning: true      # lessons from your corrections and verified results
        calibration: true   # the track record of claimed results, and what it changes
```

**Lessons.** While there is work going on (a task in focus, or tools used in the last turn), your
own words teach:

- "以後跑測試都要用 uv run pytest" / "from now on, run the tests with …" holds at once, for this
  project; for every project only when you say so.
- A correction ("不對，要用 uv run pytest") is kept on trial. It is put in force when you repeat it in
  another conversation or confirm it ("對，以後都這樣"), or, for a command, after it was followed
  and worked in two tasks: the model ran that command itself (printing it, or `! cmd`, does not
  count) and the run backs a result that held. Said again word for word (with the same exception)
  it counts once more; any other wording — a turned direction, a moved "not", another exception —
  replaces the old lesson.
- "那條規則不對，忘掉它" / "forget that rule" withdraws the lesson just shown. "Forget the rule about
  ruff" (or its id) withdraws that one, even if another was shown; a name that matches no lesson of
  this project is asked about, never guessed. When it is not clear which lesson you mean, the model
  may settle one of the lessons you were shown — once — or ask; it cannot use your "yes" for a
  lesson it proposes itself. Confirming works the same way.
- Words that would drop a check or an approval ("以後推 repo 不用問", "deploy without approval") are
  never learned, from you or from the model — they are a permission, and permissions stay with the
  task's authorization rules. A stricter wish ("never deploy without approval") is kept. Saying an
  answer was wrong teaches nothing by itself; a correction says how you want the work done, never
  that a technical claim is true. The model may propose a lesson; it stays on trial until you
  confirm it.

A lesson rides the current user message (never the system prompt) as a dated, advisory
`<lessons>` block — not an instruction or a permission: your latest message, recorded limits and
stopped tasks always win. It is shown only for this project, for the kind of work it came from,
when the request or task shares its words, and not to a model or version it was not learned for.
A withdrawn, replaced or lapsed lesson is never shown again. Plain chat gets nothing. Lessons that
do not fit `lessons_max_chars` are left out whole, never cut short, and only the ones shown count as
shown (for a confirmation, or for the credit of following them).

**Track record.** Each state claim the model makes is recorded with how it held up (backed by our
own check, failed, or unverified; later corrected or confirmed by you), per task kind, model,
provider and Hermes version — never pooled. A subagent's or coding agent's report counts for that
agent as "reported" until our own check, after the report, of what the report names confirms or
refutes it: a check of `test_b` says nothing about a report on `test_a`, a report on both needs
both, and a check made before the report says nothing about it. A command that only prints or
searches (`echo 'uv run pytest'`) is not a test run, and `! cmd` does not show that `cmd` passed.
The same check run again by the same model counts once, on its first record, and your correction
lands there; another model's rerun, or the same command on another file, port or id, is its own
record. With fewer than
`calibration_min_samples` results it says nothing. When the record for the kind of work in focus
is poor, the turn gets one line ("check it and read the result before you say so") and the full
answer check (one auxiliary call); a good record never relaxes anything, and chat outside a task
is never checked harder.

```bash
hermes cognition lessons                 # this project's lessons; show / confirm / revoke / forget ID
hermes cognition track-record            # results per task kind, model and version; --reset clears
```

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `cognition.evidence_loop.max_rounds` | `1` | check rounds per turn |
| `cognition.evidence_loop.max_extra_tool_calls` | `3` | calls in a check round |
| `cognition.consistency_guard.evidence_context` | `true` | the deep-route verifier sees the turn's tool results |
| `plugins.entries.cognition_wm.settings.snapshot_max_chars` | `1600` | largest snapshot |
| `plugins.entries.cognition_wm.settings.other_open_tasks_max` | `3` | other open tasks listed |
| `plugins.entries.cognition_wm.settings.authorization_historical_after_hours` | `12` | when a permission must be confirmed again |
| `plugins.entries.cognition_wm.settings.scope_gate` | `true` | keep execution tools inside what the user asked for |
| `plugins.entries.cognition_wm.settings.cancelled_under_yolo` | `block` | `approve` lets `/yolo` pass stopped tasks |
| `plugins.entries.cognition_wm.settings.learning` | `false` | learn and recall lessons |
| `plugins.entries.cognition_wm.settings.calibration` | `false` | keep the track record and let a poor one add a check |
| `plugins.entries.cognition_wm.settings.lessons_max` | `3` | lessons added to a message |
| `plugins.entries.cognition_wm.settings.lessons_max_chars` | `600` | largest lessons block |
| `plugins.entries.cognition_wm.settings.calibration_min_samples` | `5` | results before a track record counts |
| `plugins.entries.cognition_wm.settings.expire_after_days` | `30` | an unconfirmed lesson with no support expires |
| `plugins.entries.cognition_wm.settings.stale_after_days` | `90` | a lesson with no support goes stale |

Lapsing, and the retention of evidence and results, is applied when a process first uses a profile and
then at most once an hour, so a long-running CLI keeps these limits without a restart.

## Turn it off

Remove `cognition_wm` from `plugins.enabled` and set `evidence_loop.enabled: false`. The state
stays in `<hermes home>/plugin-data/cognition_wm/working_memory.db` and affects nothing else;
deleting that file removes it completely.

Setting `learning` or `calibration` back to `false` stops lessons or the track record from the next
turn on; `hermes cognition lessons forget --all` and `hermes cognition track-record --reset` clear
them without touching the working memory.

## Limits

- A task, limit or permission counts only once the model records it; the snapshot and the
  approval prompt ask it to.
- Command reading is heuristic and never runs anything: code imported by a one-liner, or a command
  deliberately disguised (built from pieces at run time), can slip past it; the approval rules
  still apply to such calls.
- Opening a web page counts as reading; a page whose mere visit changes something is not caught.
- Lessons and the track record read words, not meaning: a lesson whose wording shares nothing
  with the request is not recalled, and a question that sounds like a correction ("錯了嗎？")
  can count as one. Whether a real model follows a lesson is measured by the evaluation, not
  assumed.
- Lessons are kept only in this profile's working-memory database. A memory or skill write that
  carries their markers or repeats a lesson word for word — one in force, or one withdrawn,
  replaced or lapsed that is still kept as history — is refused, but the background memory
  review sees earlier turns as they were sent (lessons included, to keep the prompt cache), so a
  reworded copy can still be saved; revoking or forgetting a lesson does not remove such a copy.
- A report that names nothing in particular ("all tests pass") is settled by our own check of that
  kind of claim, by the rule our own claims are held to.
