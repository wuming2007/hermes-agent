"""Cognition v2 phase 2 in the turn: learn from the owner's words, keep the track record, and credit
the lessons that were followed and worked.

``before_turn`` runs in the plugin's ``pre_llm_call``, ``after_turn`` in its ``post_llm_call``. Each is
fail-open on its own: an error leaves the turn without lessons or a record and never takes the
working memory with it. A new lesson is learned only while there is work going on (a focus task, or
tools used in the last turn): plain chat teaches nothing, and a wish about the user themselves
belongs to long-term memory, not here. A confirmation or a withdrawal is heard whenever the owner
says it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from plugins.cognition_wm import calibration, outcomes, recall, runtime
from plugins.cognition_wm.experience_store import ExperienceStore
from plugins.cognition_wm.lessons import (
    LessonCue,
    correction_kind,
    draft_lesson,
    read_cue,
    same_topic,
    same_wording,
    search_terms,
    work_kind,
)
from plugins.cognition_wm.model import Experience, SourceRef, Task, clip
from plugins.cognition_wm.provenance import RUN_ID, turn_source

logger = logging.getLogger(__name__)
LAST_TURN_KEY = "last_turn:{}"  # lineage -> the turn whose outcomes a correction marks
EXPOSED_KEY = "exposed:{}"  # lineage -> the lessons shown in its latest turn
_LIVE = ("candidate", "active", "stale")
_CHAIN_DEPTH = 5


@dataclass(frozen=True)
class TurnLessons:
    """What phase 2 adds to a turn: context for the user message, the text for the answer check,
    and ``turn_guidance`` for the core."""

    context: str = ""
    verifier_body: str = ""
    guidance: dict = field(default_factory=dict)


def _digest(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]


def _meta(wm: ExperienceStore, key: str) -> dict:
    raw = wm.get_meta(key)
    return json.loads(raw) if raw else {}


def _owner(state: runtime.TurnState) -> SourceRef:
    return turn_source(owner=True, session_id=state.session_id, turn_id=state.turn_id)


def _checked_by_us(state: runtime.TurnState, ref: str) -> SourceRef:
    return SourceRef(type="tool_result", session_id=state.session_id, turn_id=state.turn_id, ref=ref, at=time.time(),
                     authored_by="tool", run_id=RUN_ID)


def _calls(message: Any) -> list[tuple[str, dict]]:
    """(tool name, arguments) of an assistant row's calls, looking inside the ``tool_call`` wrapper."""
    from utils import safe_json_loads

    found = []
    for call in (message.get("tool_calls") or []) if message.get("role") == "assistant" else []:
        function = call.get("function") or {}
        args = safe_json_loads(function.get("arguments") or "{}") or {}
        if function.get("name") == "tool_call" and isinstance(args.get("calls"), list):
            found += [(str(c.get("name") or ""), c.get("arguments") or {}) for c in args["calls"] if isinstance(c, dict)]
        else:
            found.append((str(function.get("name") or ""), args if isinstance(args, dict) else {}))
    return found


_COMMAND_DIRS = re.compile(r"^\S*/(?=\S)")


def last_turn_steps(history: Sequence[Any]) -> tuple[str, ...]:
    """The tool calls of the turn before this one ("terminal: pytest -q"), oldest first."""
    rows = [m for m in history or () if isinstance(m, dict)]
    users = [i for i, m in enumerate(rows) if m.get("role") == "user"]
    if len(users) < 2:
        return ()
    steps = []
    for message in rows[users[-2]:users[-1]]:
        for name, args in _calls(message):
            target = (_COMMAND_DIRS.sub("", str(args["command"])) if args.get("command") else
                      next((str(args[k]) for k in ("path", "code", "url", "query") if args.get(k)), ""))
            steps.append(clip(f"{name}: {target}" if target else name, 80))
    return tuple(steps)


def last_request(history: Sequence[Any]) -> str:
    """What the user asked in the turn before this one (not a runtime note)."""
    users = [m for m in history or () if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) < 2:
        return ""
    content = users[-2].get("content")
    text = " ".join((" ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
                     if isinstance(content, list) else str(content or "")).split())
    return "" if text.startswith("[") else clip(text, 200)


def _voiced(wm: ExperienceStore, lesson: Optional[Experience]) -> frozenset:
    """The conversations in which the user spoke for a lesson or the ones it replaced."""
    lineages: set[str] = set()
    for _ in range(_CHAIN_DEPTH):
        if lesson is None:
            break
        lineages |= wm.signal_summary(lesson.experience_id).user_lineages
        lesson = wm.experience(lesson.supersedes) if lesson.supersedes else None
    return frozenset(lineages)


def _first_status(kind: str, match: Optional[Experience], voiced: frozenset, lineage: str) -> str:
    """A standing wish holds at once; a correction is tried first, unless it revises a lesson in
    force or the user already said it in another conversation."""
    if kind == "directive":
        return "active"
    if match is None:
        return "candidate"
    return "active" if match.status == "active" or voiced - {lineage} else "candidate"


def _put_in_force_if_voiced_elsewhere(wm: ExperienceStore, lesson: Experience, owner: SourceRef, lineage: str) -> None:
    if lesson.status == "stale" or (lesson.status == "candidate" and _voiced(wm, lesson) - {lineage}):
        wm.set_experience_status(lesson.experience_id, "active", source=owner)


def _state_lesson(wm: ExperienceStore, state: runtime.TurnState, cue: LessonCue, *, focus: Optional[Task],
                  steps: Sequence[str], request: str, shown: Sequence[str], model: str, provider: str) -> list[str]:
    """A directive or correction: said again, it counts once more; revised, it replaces the old one."""
    if cue.loosens:
        return []  # a permission, never a lesson: the task's authorization rules keep it
    owner = _owner(state)
    source_class = "user_directive" if cue.kind == "directive" else "user_correction"
    key = f"msg:{state.lineage_root}:{_digest(cue.text)}"
    draft = draft_lesson(cue, focus_title=focus.title if focus else "", focus_kind=focus.kind if focus else None,
                         previous_steps=steps, previous_request=request)
    topic = frozenset(draft.lesson_key.split())
    live = [e for e in wm.experiences(project_key=state.project_key, statuses=_LIVE,
                                      include_global=draft.scope == "global") if e.scope == draft.scope]
    match = next((e for e in reversed(live) if same_topic(topic, frozenset(e.lesson_key.split()))), None)
    if match is not None and same_wording(match.behavior, draft.behavior):
        wm.add_signal(match.experience_id, source_class=source_class, polarity=1, evidence_key=key, source=owner,
                      lineage=state.lineage_root, detail=cue.text)
        _put_in_force_if_voiced_elsewhere(wm, match, owner, state.lineage_root)
        return []
    lesson = wm.add_experience(
        kind=draft.kind, scope=draft.scope, project_key=state.project_key,
        task_kind=focus.kind if focus else work_kind(request, steps),
        situation=draft.situation, behavior=draft.behavior, exceptions=draft.exceptions, tags=draft.tags,
        lesson_key=draft.lesson_key, status=_first_status(cue.kind, match, _voiced(wm, match), state.lineage_root),
        origin=owner, learned_under=runtime.environment(model, provider), supersedes=match.experience_id if match else None)
    wm.add_signal(lesson.experience_id, source_class=source_class, polarity=1, evidence_key=key, source=owner,
                  lineage=state.lineage_root, detail=cue.text)
    if cue.kind == "correction":  # it also speaks against a lesson shown last turn that it touches
        words = set(draft.tags)
        for other in filter(None, (wm.experience(i) for i in shown)):
            if other.status in _LIVE and other.experience_id != lesson.experience_id and words & set(other.tags):
                wm.add_signal(other.experience_id, source_class="user_correction", polarity=-1, evidence_key=key,
                              source=owner, lineage=state.lineage_root, detail=cue.text)
    return []


def _spoken_here(wm: ExperienceStore, state: runtime.TurnState, statuses: Sequence[str]) -> list[Experience]:
    """Lessons of this project that came up in this conversation: the user spoke for them here, or
    the model proposed them here."""
    found = []
    for lesson in wm.experiences(project_key=state.project_key, statuses=statuses):
        voiced_here = state.lineage_root in wm.signal_summary(lesson.experience_id).user_lineages
        if voiced_here or runtime.lineage_root(lesson.origin.session_id) == state.lineage_root:
            found.append(lesson)
    return found


def _confirm_lesson(wm: ExperienceStore, state: runtime.TurnState, cue: LessonCue, *, shown: Sequence[str],
                    **_: Any) -> list[str]:
    """The owner confirms the tentative lesson just shown, else the latest one from this conversation."""
    state.notes["lesson_intent"] = "confirmation"
    tentative = [e for e in filter(None, (wm.experience(i) for i in shown)) if e.status == "candidate"]
    tentative = tentative or _spoken_here(wm, state, ("candidate",))
    if not tentative:
        return []
    lesson = max(tentative, key=lambda e: e.created_at)
    owner = _owner(state)
    wm.add_signal(lesson.experience_id, source_class="user_confirmation", polarity=1,
                  evidence_key=f"msg:{state.lineage_root}:{_digest(cue.text)}", source=owner,
                  lineage=state.lineage_root, detail=cue.text)
    wm.set_experience_status(lesson.experience_id, "active", source=owner)
    return []


def _revoke_lesson(wm: ExperienceStore, state: runtime.TurnState, cue: LessonCue, *, shown: Sequence[str],
                   **_: Any) -> list[str]:
    """The owner withdraws a lesson: the one just shown (or the one their words name), else the only
    one from this conversation. When it cannot be told which, the model is asked to name it."""
    state.notes["lesson_intent"] = "revoke"
    live = [e for e in filter(None, (wm.experience(i) for i in shown)) if e.status in _LIVE]
    words = set(search_terms(cue.text))
    named = [e for e in live if words & set(e.tags)]
    target = live[0] if len(live) == 1 else named[0] if len(named) == 1 else None
    if target is None and not live:
        here = _spoken_here(wm, state, _LIVE)
        target = here[0] if len(here) == 1 else None
    if target is None:
        return [f'Pending: the user asked to drop a lesson ("{clip(cue.text, 120)}"). If it is one of these, '
                'revoke it with task_state (action "lesson", op "revoke", experience_id).']
    wm.set_experience_status(target.experience_id, "revoked", source=_owner(state))
    return []


# cue kind -> what learning does with it (a result confirmation only counts in the track record)
_CUE_ACTIONS: dict[str, Callable[..., list[str]]] = {
    "directive": _state_lesson, "correction": _state_lesson, "confirmation": _confirm_lesson,
    "revoke": _revoke_lesson, "result_confirmation": lambda *_a, **_k: [],
}


def _relabel_last_turn(wm: ExperienceStore, state: runtime.TurnState, last: dict, cue: Optional[LessonCue],
                       text: str) -> None:
    """The owner's word on the last answer. Saying a result did not hold marks our claims there
    corrected; correcting how it was done leaves what they read and counts as the user's effort; a
    confirmed result makes the unknown or reported ones there a success."""
    there = wm.outcomes(session_id=last.get("session"), turn_id=last.get("turn"))
    ours = [o.outcome_id for o in there if o.actor_kind == "self" and o.result not in ("corrected", "blocked")]
    corrected = correction_kind(cue, text)
    if corrected == "result":
        wm.relabel_outcomes(ours, "corrected", source=_owner(state), correction=True)
    elif corrected == "approach":
        wm.note_corrections(ours)
    elif cue is not None and cue.kind == "result_confirmation":
        wm.relabel_outcomes([o.outcome_id for o in there if o.result in ("unknown", "reported")], "success",
                            source=_owner(state))


def before_turn(state: runtime.TurnState, text: str, history: Sequence[Any], *, model: str,
                provider: str) -> TurnLessons:
    current = runtime.settings()
    if not (current.learning or current.calibration):
        return TurnLessons()
    try:
        return _before_turn(state, text, history, current=current, model=model, provider=provider)
    except Exception as exc:  # phase 2 must never break a turn or the working memory
        logger.warning("cognition_wm learning failed before the turn (no lessons this turn): %s", exc)
        return TurnLessons()


def _before_turn(state: runtime.TurnState, text: str, history: Sequence[Any], *, current: runtime.Settings,
                 model: str, provider: str) -> TurnLessons:
    wm = runtime.store()
    focus = runtime.focus_task(state)
    notes: list[str] = []
    if state.owner:
        shown = _meta(wm, EXPOSED_KEY.format(state.lineage_root)).get("ids") or []
        # A stop, pause or resume is about this task, never a lesson; a limit may also be a standing wish.
        cue = read_cue(text) if state.intent is None or state.intent.kind == "restrict" else None
        last = _meta(wm, LAST_TURN_KEY.format(state.lineage_root))
        if current.calibration and last:
            _relabel_last_turn(wm, state, last, cue, text)
        if current.learning and cue is not None:
            steps = last_turn_steps(history)
            if focus is not None or steps or cue.kind in ("confirmation", "revoke"):
                notes = _CUE_ACTIONS[cue.kind](wm, state, cue, focus=focus, steps=steps, request=last_request(history),
                                               shown=shown, model=model, provider=provider)
    lessons: list[Experience] = []
    if current.learning and state.owner:
        lessons = recall.select(wm, state, text=text, focus=focus, model=model, provider=provider,
                                limit=int(current.lessons_max))
        wm.set_meta(EXPOSED_KEY.format(state.lineage_root),
                    json.dumps({"turn": state.turn_id, "ids": [lesson.experience_id for lesson in lessons]}))
    guidance: dict = {}
    record_lines: list[str] = []
    if current.calibration and focus is not None:  # plain chat is never checked harder
        guidance, record_lines = calibration.for_turn(wm, state, focus, model=model, provider=provider,
                                                      min_samples=int(current.calibration_min_samples))
    body = recall.render_body(wm, lessons, record_lines=record_lines, notes=notes,
                              max_chars=int(current.lessons_max_chars))
    if not body:
        return TurnLessons(guidance=guidance)
    return TurnLessons(context=recall.fence(body, history), verifier_body=f"{recall.VERIFIER_NOTE}\n{body}",
                       guidance=guidance)


def after_turn(*, session_id: str, turn_id: str, assistant_response: Any, history: Sequence[Any], model: str,
               provider: str) -> None:
    current = runtime.settings()
    if not (current.learning or current.calibration):
        return
    try:
        state = runtime.current_turn(session_id)
        if state is not None and state.turn_id == (turn_id or state.turn_id):
            _after_turn(state, str(assistant_response or ""), history, current=current, model=model, provider=provider)
    except Exception as exc:  # phase 2 must never break a turn or the working memory
        logger.warning("cognition_wm learning failed after the turn (nothing recorded): %s", exc)


def on_post_llm_call(*, session_id: str = "", turn_id: str = "", assistant_response: Any = None,
                     conversation_history: Any = None, model: str = "", provider: str = "", **_: Any) -> None:
    after_turn(session_id=session_id or "", turn_id=turn_id or "", assistant_response=assistant_response,
               history=conversation_history or [], model=model or "", provider=provider or "")


def _done_this_turn(wm: ExperienceStore, state: runtime.TurnState) -> list[Task]:
    return [t for t in wm.tasks(project_key=state.project_key, lineage_root=state.lineage_root, open_only=False)
            if t.status == "done" and (t.status_source.session_id, t.status_source.turn_id) == (state.session_id,
                                                                                               state.turn_id)]


def _after_turn(state: runtime.TurnState, answer: str, history: Sequence[Any], *, current: runtime.Settings,
                model: str, provider: str) -> None:
    from agent.cognition_evidence import build_evidence_report

    wm = runtime.store()
    report = build_evidence_report(answer, list(history or []))
    finished = _done_this_turn(wm, state)
    task = runtime.focus_task(state) or (finished[0] if finished else None)
    scope_key = f"task:{task.task_id}" if task else f"conv:{state.lineage_root}"
    drafts = outcomes.turn_outcomes(report, model=model, scope_key=scope_key,
                                    turn_key=f"{state.session_id}:{state.turn_id}")
    if current.calibration:
        _record(wm, state, drafts, task=task, finished=finished, model=model, provider=provider)
        wm.set_meta(LAST_TURN_KEY.format(state.lineage_root),
                    json.dumps({"session": state.session_id, "turn": state.turn_id}))
    if current.learning:
        _credit_followed(wm, state, report, drafts, scope_key)


def _record(wm: ExperienceStore, state: runtime.TurnState, drafts: Sequence[outcomes.OutcomeDraft], *,
            task: Optional[Task], finished: Sequence[Task], model: str, provider: str) -> None:
    version = runtime.environment(model, provider)["hermes_version"]
    asked = {"asks": int(state.notes.get("asks", 0)), "blocks": int(state.notes.get("blocks", 0))}
    common = dict(session_id=state.session_id, turn_id=state.turn_id, project_key=state.project_key,
                  provider=provider or "", hermes_version=version)
    for draft in drafts:
        wm.record_outcome(evidence_key=draft.key, task_id=task.task_id if task else None,
                          task_kind=task.kind if task else "none", claim_class=draft.claim_class,
                          actor_kind=draft.actor_kind, actor_id=draft.actor_id, result=draft.result, reason=draft.reason,
                          evidence_ids=draft.evidence_ids, intervention=asked if draft.actor_kind == "self" else {},
                          **common)
    for done in finished:  # a task marked done: backed only when each deliverable cites our own checks
        verified = wm.verified_evidence_ids(done.task_id)
        deliverables = wm.items(done.task_id, kinds=("deliverable",))
        backed = bool(deliverables) and all(
            item.state == "dropped" or (item.state == "done" and set(item.evidence_ids) & verified) for item in deliverables)
        wm.record_outcome(evidence_key=f"task:{done.task_id}:done", task_id=done.task_id, task_kind=done.kind,
                          claim_class="task", actor_kind="self", actor_id=model or "unknown-model",
                          result="success" if backed else "unknown", reason="" if backed else "done_without_evidence",
                          intervention={**asked, "user_turns": wm.owner_turns(done.task_id)}, **common)
    # our own check settles what a subagent or coding agent reported earlier in the same work
    ours = outcomes.own_verdicts(drafts)
    if ours:
        earlier = wm.outcomes(task_id=task.task_id) if task else wm.outcomes(session_id=state.session_id)
        for claim_class, verdict in ours.items():
            wm.relabel_outcomes([o.outcome_id for o in earlier if o.actor_kind != "self" and o.claim_class == claim_class
                                 and o.result == "reported"], verdict, source=_checked_by_us(state, claim_class))


def _credit_followed(wm: ExperienceStore, state: runtime.TurnState, report: Any,
                     drafts: Sequence[outcomes.OutcomeDraft], scope_key: str) -> None:
    """A lesson shown this turn, followed by a command of ours that worked: one count per task."""
    shown = _meta(wm, EXPOSED_KEY.format(state.lineage_root))
    if shown.get("turn") != state.turn_id:
        return
    if any(d.actor_kind == "self" and d.result in ("failure", "partial") for d in drafts):
        return
    for lesson in filter(None, (wm.experience(i) for i in shown.get("ids") or ())):
        hits = outcomes.followed(outcomes.command_markers(lesson.behavior), report.evidence)
        if lesson.status not in _LIVE or not hits:
            continue
        checked = _checked_by_us(state, hits[-1].evidence_id)
        if not wm.add_signal(lesson.experience_id, source_class="tool_verified", polarity=1, evidence_key=scope_key,
                             source=checked, lineage=state.lineage_root, detail=hits[-1].target):
            continue
        summary = wm.signal_summary(lesson.experience_id)
        promote = (lesson.status == "candidate" and lesson.kind in ("procedure", "fact")
                   and summary.verifications >= 2 and not summary.corrections)
        if promote or lesson.status == "stale":
            wm.set_experience_status(lesson.experience_id, "active", source=checked)
