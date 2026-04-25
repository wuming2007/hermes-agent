import json

from agent.cognition_trace_report import (
    TRACE_REPORT_SCHEMA_VERSION,
    analyze_cognition_trace_entries,
    analyze_cognition_trace_jsonl,
    empty_cognition_trace_report,
)


def _entry(trace=None, completed=True):
    entry = {"conversations": [], "model": "test-model"}
    if completed is not None:
        entry["completed"] = completed
    if trace is not None:
        entry["metadata"] = {"cognition_trace": trace}
    return entry


def _trace(**overrides):
    trace = {
        "schema_version": 1,
        "enabled": True,
        "route": {
            "mode": "standard",
            "original_mode": "fast",
            "retrieval_plan": "standard",
            "verification_plan": "light",
            "allow_cheap_model": False,
            "consistency_check": True,
        },
        "uncertainty": {
            "present": True,
            "confidence_band": "low",
            "action": "escalate_depth",
            "depth_escalated": True,
            "require_tool_evidence": True,
            "seek_human": False,
            "target_mode": "deep",
        },
        "verification": {
            "ladder_enabled": True,
            "ladder_source_plan": "full",
            "ladder_stages": ["self_correction", "fast_monitor", "slow_verifier"],
            "ladder_applied_stages": ["self_correction", "slow_verifier"],
            "applied": True,
            "changed": False,
        },
    }
    trace.update(overrides)
    return trace


def test_empty_report_has_stable_zero_shape():
    report = empty_cognition_trace_report(files=["trajectory_samples.jsonl"])

    assert report["schema_version"] == TRACE_REPORT_SCHEMA_VERSION
    assert report["files"] == ["trajectory_samples.jsonl"]
    assert report["total_entries"] == 0
    assert report["completed"] == {"true": 0, "false": 0, "missing": 0}
    assert report["cognition_trace"]["present"] == 0
    assert report["cognition_trace"]["missing"] == 0
    assert report["cognition_trace"]["malformed"] == 0
    assert report["route"]["modes"] == {}
    assert report["errors"] == {"malformed_jsonl": 0, "missing_files": 0}


def test_analyze_entries_counts_completion_and_trace_presence():
    report = analyze_cognition_trace_entries([
        _entry(_trace(), completed=True),
        _entry(completed=False),
        _entry(completed=None),
        {"metadata": {"cognition_trace": "not-a-dict"}},
    ])

    assert report["total_entries"] == 4
    assert report["completed"] == {"true": 1, "false": 1, "missing": 2}
    assert report["cognition_trace"]["present"] == 1
    assert report["cognition_trace"]["missing"] == 2
    assert report["cognition_trace"]["malformed"] == 1
    assert report["cognition_trace"]["enabled"] == {"true": 1, "false": 0, "missing": 0}
    assert report["cognition_trace"]["schema_versions"] == {"1": 1}


def test_analyze_entries_counts_route_uncertainty_and_verification():
    report = analyze_cognition_trace_entries([
        _entry(_trace()),
        _entry(
            _trace(
                enabled=False,
                route={"mode": "disabled", "allow_cheap_model": None, "consistency_check": None},
                uncertainty={"present": False},
                verification={"ladder_enabled": False, "ladder_stages": ["self_correction"]},
            )
        ),
    ])

    assert report["route"]["modes"] == {"standard": 1, "disabled": 1}
    assert report["route"]["original_modes"] == {"fast": 1}
    assert report["route"]["retrieval_plans"] == {"standard": 1}
    assert report["route"]["verification_plans"] == {"light": 1}
    assert report["route"]["allow_cheap_model"] == {"true": 0, "false": 1, "missing": 1}
    assert report["route"]["consistency_check"] == {"true": 1, "false": 0, "missing": 1}

    assert report["uncertainty"]["present"] == {"true": 1, "false": 1, "missing": 0}
    assert report["uncertainty"]["confidence_bands"] == {"low": 1}
    assert report["uncertainty"]["actions"] == {"escalate_depth": 1}
    assert report["uncertainty"]["depth_escalated"] == {"true": 1, "false": 0, "missing": 1}
    assert report["uncertainty"]["require_tool_evidence"] == {"true": 1, "false": 0, "missing": 1}
    assert report["uncertainty"]["seek_human"] == {"true": 0, "false": 1, "missing": 1}
    assert report["uncertainty"]["target_modes"] == {"deep": 1}

    assert report["verification"]["ladder_enabled"] == {"true": 1, "false": 1, "missing": 0}
    assert report["verification"]["ladder_source_plans"] == {"full": 1}
    assert report["verification"]["ladder_stages"] == {
        "self_correction": 2,
        "fast_monitor": 1,
        "slow_verifier": 1,
    }
    assert report["verification"]["ladder_applied_stages"] == {
        "self_correction": 1,
        "slow_verifier": 1,
    }
    assert report["verification"]["applied"] == {"true": 1, "false": 0, "missing": 1}
    assert report["verification"]["changed"] == {"true": 0, "false": 1, "missing": 1}


def test_analyze_jsonl_counts_malformed_lines_and_missing_files(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text(
        json.dumps(_entry(_trace(route={"mode": "deep"})), ensure_ascii=False)
        + "\n"
        + "{bad json\n"
        + json.dumps(_entry(), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    report = analyze_cognition_trace_jsonl([path, tmp_path / "missing.jsonl"])

    assert report["files"] == [str(path), str(tmp_path / "missing.jsonl")]
    assert report["total_entries"] == 2
    assert report["route"]["modes"] == {"deep": 1}
    assert report["cognition_trace"]["present"] == 1
    assert report["cognition_trace"]["missing"] == 1
    assert report["errors"] == {"malformed_jsonl": 1, "missing_files": 1}
