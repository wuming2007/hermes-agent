"""Working memory for Hermes (cognition v2): task state, the user's limits and cancellations, what
the user allowed, and tool evidence, kept across turns, resumed chats and compaction.

Enable with ``plugins.enabled: [cognition_wm]``. Settings: ``cognition.working_memory`` in
config.yaml. The snapshot rides the current user message (``pre_llm_call``); the system prompt
only gains a fixed usage note at session start, so the prompt cache prefix never changes.
"""

from __future__ import annotations

from plugins.cognition_wm import evidence_link, scope_gate, tool, turn

USAGE = (
    "Working memory: Hermes keeps the state of the user's work — tasks, decisions, limits, "
    "cancellations and what the user allowed — and shows it at the end of the user's message inside "
    "<working-memory> tags. It is bookkeeping, not the user speaking. A PAUSED or CANCELLED task stays "
    "stopped until the user asks again in their own words. When the user starts a piece of work, "
    "changes its scope, allows or forbids an action, pauses or cancels it, record that with the "
    "task_state tool. Mark a deliverable done only with evidence ids from tool results."
)


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", turn.on_pre_llm_call)
    ctx.register_hook("post_tool_call", evidence_link.on_post_tool_call)
    ctx.register_hook("pre_tool_call", scope_gate.on_pre_tool_call)
    ctx.register_tool(name="task_state", toolset="cognition_wm", schema=tool.SCHEMA, handler=tool.handle,
                      description=tool.SCHEMA["description"], emoji="🧭")
    ctx.register_system_prompt_section("cognition-wm", USAGE, position="after_memory", max_chars=len(USAGE) + 100)
