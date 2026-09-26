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

## Turn it off

Remove `cognition_wm` from `plugins.enabled` and set `evidence_loop.enabled: false`. The state
stays in `<hermes home>/plugin-data/cognition_wm/working_memory.db` and affects nothing else;
deleting that file removes it completely.

## Limits

- A task, limit or permission counts only once the model records it; the snapshot and the
  approval prompt ask it to.
- Command reading is heuristic and never runs anything: code imported by a one-liner, or a command
  deliberately disguised (built from pieces at run time), can slip past it; the approval rules
  still apply to such calls.
- Opening a web page counts as reading; a page whose mere visit changes something is not caught.
