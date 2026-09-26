"""The track record in the turn (cognition v2, phase 2): how this model's claimed results went for
this kind of work, and what that changes.

Results are kept apart per task kind, claim, executor, model, provider and Hermes version. This
project's record is read first; every project's only when this one has too few results. Below
``calibration_min_samples`` it says nothing and changes nothing. A poor record adds one line ("check
before you say so") and raises this turn's answer check to the full verifier; a good record changes
nothing, since it never relaxes a check. It may point at an executor that did better here, as a
suggestion only, and never starts one. The estimate is the lower end of an 80% Wilson interval, so a
handful of lucky results never reads as a sure thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from plugins.cognition_wm import runtime
from plugins.cognition_wm.experience_store import ExperienceStore
from plugins.cognition_wm.model import Outcome, Task
from plugins.cognition_wm.provenance import RUN_ID

Z = 1.2816  # one-sided 80%
STEADY_SAMPLES = 15
# Results that say how a claim held up; blocked (the user or a rule stopped it) and reported (someone
# else's word) do not.
_COUNTED = frozenset({"success", "failure", "partial", "unknown", "corrected"})
_BAND_ORDER = ("low", "medium", "high", "unknown")
_CLAIM_WORDS = {"tests_pass": "tests pass", "deployed": "deployed", "verified": "verified", "working": "it works",
                "written": "written", "completed": "done", "task": "task done"}


def wilson_lower(successes: int, n: int, z: float = Z) -> float:
    """The lower end of the Wilson interval for ``successes`` out of ``n``."""
    if n <= 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, min(p, (centre - margin) / (1 + z * z / n)))


def band_for(results: Sequence[str], *, min_samples: int) -> str:
    """unknown (too few results), low, medium or high."""
    counted = [result for result in results if result in _COUNTED]
    if len(counted) < max(1, min_samples):
        return "unknown"
    lower = wilson_lower(counted.count("success"), len(counted))
    if lower < 0.5 or counted[-10:].count("corrected") >= 2:
        return "low"
    return "medium" if lower < 0.7 else "high"


@dataclass(frozen=True)
class Cell:
    """The results of one kind of claim, by one executor, model, provider and Hermes version."""

    task_kind: str
    claim_class: str
    actor_kind: str
    actor_id: str
    provider: str
    hermes_version: str
    results: tuple[str, ...]  # oldest first

    @property
    def counted(self) -> list[str]:
        return [result for result in self.results if result in _COUNTED]

    @property
    def verified(self) -> int:
        return self.counted.count("success")

    @property
    def lower(self) -> float:
        return wilson_lower(self.verified, len(self.counted))

    def band(self, min_samples: int) -> str:
        return band_for(self.results, min_samples=min_samples)


def cells(found: Iterable[Outcome]) -> list[Cell]:
    grouped: dict[tuple, list[str]] = {}
    for outcome in found:
        key = (outcome.task_kind, outcome.claim_class, outcome.actor_kind, outcome.actor_id, outcome.provider,
               outcome.hermes_version)
        grouped.setdefault(key, []).append(outcome.result)
    return [Cell(*key, results=tuple(results)) for key, results in grouped.items()]


def _record_line(cell: Cell, scope: str, model: str) -> str:
    corrected = cell.results[-10:].count("corrected")
    said = f'"{_CLAIM_WORDS.get(cell.claim_class, cell.claim_class)}" {cell.verified} verified of {len(cell.counted)}'
    said += f", {corrected} later corrected" if corrected else ""
    said += " (small sample)" if len(cell.counted) < STEADY_SAMPLES else ""
    return (f"Track record ({scope} · {cell.task_kind} · {model} · Hermes {cell.hermes_version}): {said} → check it "
            "and read the result before you say so.")


def _delegation_line(wm: ExperienceStore, state: runtime.TurnState, focus: Task, *, min_samples: int) -> list[str]:
    """Another executor that did this work well here: named, never started from here."""
    others = [cell for cell in cells(wm.outcomes(project_key=state.project_key, task_kind=focus.kind))
              if cell.actor_kind != "self" and cell.band(min_samples) == "high"]
    if not others:
        return []
    best = max(others, key=lambda cell: cell.lower)
    record = f"{best.actor_id} did this kind of work well here ({best.verified} checked of {len(best.counted)})"
    if wm.effective_authorizations(focus.task_id, run_id=RUN_ID).get("coding_agent_launch") == "granted":
        return [f"{record}; the user allowed delegating for this task."]
    return [f"{record}; you may suggest delegating to the user, and not start it without their go-ahead."]


def for_turn(wm: ExperienceStore, state: runtime.TurnState, focus: Task, *, model: str, provider: str,
             min_samples: int) -> tuple[dict, list[str]]:
    """(``turn_guidance`` for the core, lines for the lessons block) for a turn with a task in focus."""
    version = runtime.environment(model, provider)["hermes_version"]
    mine = dict(task_kind=focus.kind, actor_kind="self", actor_id=model or "unknown-model", provider=provider or "",
                hermes_version=version)
    scope, found = "this project", cells(wm.outcomes(project_key=state.project_key, **mine))
    if not any(len(cell.counted) >= min_samples for cell in found):
        scope, found = "all projects", cells(wm.outcomes(**mine))
    rated = [(cell, cell.band(min_samples)) for cell in found if len(cell.counted) >= min_samples]
    if not rated:
        return {"track_record_band": "unknown"}, []
    guidance = {"track_record_band": min((band for _cell, band in rated), key=_BAND_ORDER.index)}
    weak = [cell for cell, band in rated if band == "low"]
    if not weak:
        return guidance, []
    worst = min(weak, key=lambda cell: cell.lower)
    guidance.update(verification_floor="full", reasons=[f"{focus.kind}:{worst.claim_class}"])
    return guidance, [_record_line(worst, scope, model), *_delegation_line(wm, state, focus, min_samples=min_samples)]


def describe(found: Iterable[Outcome], *, min_samples: int) -> list[str]:
    """One line per cell, for ``hermes cognition track-record``."""
    lines = []
    for cell in sorted(cells(found), key=lambda c: (c.task_kind, c.claim_class, c.actor_kind, c.actor_id, c.provider,
                                                   c.hermes_version)):
        counted = len(cell.counted)
        verdict = f"{cell.verified} verified of {counted}"
        verdict += (f" · too few results to say (needs {min_samples})" if counted < min_samples
                    else f" · lower bound {cell.lower:.2f} · {cell.band(min_samples)}")
        extras = [f"{cell.results.count(result)} {word}" for result, word in
                  (("corrected", "corrected by the user"), ("reported", "reported, not checked"),
                   ("blocked", "blocked")) if cell.results.count(result)]
        who = f"{cell.actor_kind} {cell.actor_id} ({cell.provider or '?'} · Hermes {cell.hermes_version})"
        claim = _CLAIM_WORDS.get(cell.claim_class, cell.claim_class)
        lines.append(f"{cell.task_kind} · {claim} · {who}: {verdict}" + (f" · {', '.join(extras)}" if extras else ""))
    return lines
