"""Offline cognition trace reporting for trajectory JSONL files.

PR9 intentionally stays offline-only: it reads PR8 trajectory metadata and
produces deterministic JSON-friendly counters.  It does not mutate runtime
policy, memory, prompts, or model dispatch behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

TRACE_REPORT_SCHEMA_VERSION = 1

_BOOL_BUCKETS = {"true": 0, "false": 0, "missing": 0}


def _bool_buckets() -> dict[str, int]:
    return dict(_BOOL_BUCKETS)


def empty_cognition_trace_report(files: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the stable empty PR9 cognition trace report shape."""
    return {
        "schema_version": TRACE_REPORT_SCHEMA_VERSION,
        "files": list(files or []),
        "total_entries": 0,
        "completed": _bool_buckets(),
        "cognition_trace": {
            "present": 0,
            "missing": 0,
            "enabled": _bool_buckets(),
            "schema_versions": {},
            "malformed": 0,
        },
        "route": {
            "modes": {},
            "original_modes": {},
            "retrieval_plans": {},
            "verification_plans": {},
            "allow_cheap_model": _bool_buckets(),
            "consistency_check": _bool_buckets(),
        },
        "interaction": {
            "dialogue_modes": {},
            "answer_densities": {},
            "stance_reasons": {},
        },
        "uncertainty": {
            "present": _bool_buckets(),
            "confidence_bands": {},
            "actions": {},
            "depth_escalated": _bool_buckets(),
            "require_tool_evidence": _bool_buckets(),
            "seek_human": _bool_buckets(),
            "target_modes": {},
        },
        "verification": {
            "ladder_enabled": _bool_buckets(),
            "ladder_source_plans": {},
            "ladder_stages": {},
            "ladder_applied_stages": {},
            "applied": _bool_buckets(),
            "changed": _bool_buckets(),
        },
        "errors": {"malformed_jsonl": 0, "missing_files": 0},
    }


def _increment_scalar(counter: dict[str, int], value: Any) -> None:
    if value is None:
        return
    counter[str(value)] = counter.get(str(value), 0) + 1


def _increment_bool(counter: dict[str, int], value: Any) -> None:
    if value is None:
        counter["missing"] = counter.get("missing", 0) + 1
    elif bool(value):
        counter["true"] = counter.get("true", 0) + 1
    else:
        counter["false"] = counter.get("false", 0) + 1


def _increment_list(counter: dict[str, int], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = value
    else:
        values = [value]

    for item in values:
        _increment_scalar(counter, item)


def _trace_from_entry(entry: Mapping[str, Any]) -> Any:
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    return metadata.get("cognition_trace")


def _analyze_trace(report: dict[str, Any], trace: Mapping[str, Any]) -> None:
    cognition = report["cognition_trace"]
    cognition["present"] += 1
    _increment_bool(cognition["enabled"], trace.get("enabled"))
    _increment_scalar(cognition["schema_versions"], trace.get("schema_version"))

    route = trace.get("route")
    if isinstance(route, Mapping):
        route_report = report["route"]
        _increment_scalar(route_report["modes"], route.get("mode"))
        _increment_scalar(route_report["original_modes"], route.get("original_mode"))
        _increment_scalar(route_report["retrieval_plans"], route.get("retrieval_plan"))
        _increment_scalar(route_report["verification_plans"], route.get("verification_plan"))
        _increment_bool(route_report["allow_cheap_model"], route.get("allow_cheap_model"))
        _increment_bool(route_report["consistency_check"], route.get("consistency_check"))

    interaction = trace.get("interaction")
    if isinstance(interaction, Mapping):
        interaction_report = report["interaction"]
        _increment_scalar(interaction_report["dialogue_modes"], interaction.get("dialogue_mode"))
        _increment_scalar(interaction_report["answer_densities"], interaction.get("answer_density"))
        _increment_list(interaction_report["stance_reasons"], interaction.get("stance_reasons"))

    uncertainty = trace.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        uncertainty_report = report["uncertainty"]
        _increment_bool(uncertainty_report["present"], uncertainty.get("present"))
        _increment_scalar(uncertainty_report["confidence_bands"], uncertainty.get("confidence_band"))
        _increment_scalar(uncertainty_report["actions"], uncertainty.get("action"))
        _increment_bool(uncertainty_report["depth_escalated"], uncertainty.get("depth_escalated"))
        _increment_bool(uncertainty_report["require_tool_evidence"], uncertainty.get("require_tool_evidence"))
        _increment_bool(uncertainty_report["seek_human"], uncertainty.get("seek_human"))
        _increment_scalar(uncertainty_report["target_modes"], uncertainty.get("target_mode"))

    verification = trace.get("verification")
    if isinstance(verification, Mapping):
        verification_report = report["verification"]
        _increment_bool(verification_report["ladder_enabled"], verification.get("ladder_enabled"))
        _increment_scalar(verification_report["ladder_source_plans"], verification.get("ladder_source_plan"))
        _increment_list(verification_report["ladder_stages"], verification.get("ladder_stages"))
        _increment_list(
            verification_report["ladder_applied_stages"],
            verification.get("ladder_applied_stages"),
        )
        _increment_bool(verification_report["applied"], verification.get("applied"))
        _increment_bool(verification_report["changed"], verification.get("changed"))


def _counter_dict(counter: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _sort_counter_sections(report: dict[str, Any]) -> dict[str, Any]:
    for section in ("route", "interaction", "uncertainty", "verification"):
        for key, value in list(report[section].items()):
            if isinstance(value, Counter):
                report[section][key] = _counter_dict(value)
    cognition = report["cognition_trace"]
    if isinstance(cognition.get("schema_versions"), Counter):
        cognition["schema_versions"] = _counter_dict(cognition["schema_versions"])
    return report


def analyze_cognition_trace_entries(
    entries: Iterable[Mapping[str, Any]],
    files: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyze trajectory entries containing optional metadata.cognition_trace."""
    report = empty_cognition_trace_report(files=files)

    # Use Counter while collecting scalar distributions, then convert back to
    # plain dicts for stable JSON-friendly output.
    report["cognition_trace"]["schema_versions"] = Counter()
    for section, keys in {
        "route": ("modes", "original_modes", "retrieval_plans", "verification_plans"),
        "interaction": ("dialogue_modes", "answer_densities", "stance_reasons"),
        "uncertainty": ("confidence_bands", "actions", "target_modes"),
        "verification": ("ladder_source_plans", "ladder_stages", "ladder_applied_stages"),
    }.items():
        for key in keys:
            report[section][key] = Counter()

    for entry in entries:
        report["total_entries"] += 1
        if not isinstance(entry, Mapping):
            report["completed"]["missing"] += 1
            report["cognition_trace"]["missing"] += 1
            continue

        _increment_bool(report["completed"], entry.get("completed"))
        trace = _trace_from_entry(entry)
        if trace is None:
            report["cognition_trace"]["missing"] += 1
        elif not isinstance(trace, Mapping):
            report["cognition_trace"]["malformed"] += 1
        else:
            _analyze_trace(report, trace)

    return _sort_counter_sections(report)


def analyze_cognition_trace_jsonl(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Analyze one or more trajectory JSONL files without raising on bad input."""
    path_list = [Path(path) for path in paths]
    files = [str(path) for path in path_list]
    entries: list[Mapping[str, Any]] = []
    malformed_jsonl = 0
    missing_files = 0

    for path in path_list:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        loaded = json.loads(line)
                    except json.JSONDecodeError:
                        malformed_jsonl += 1
                        continue
                    if isinstance(loaded, Mapping):
                        entries.append(loaded)
                    else:
                        malformed_jsonl += 1
        except FileNotFoundError:
            missing_files += 1

    report = analyze_cognition_trace_entries(entries, files=files)
    report["errors"]["malformed_jsonl"] = malformed_jsonl
    report["errors"]["missing_files"] = missing_files
    return report
