import json

from scripts import cognition_trace_report


def _write_entry(path, trace=None):
    entry = {"conversations": [], "model": "test", "completed": True}
    if trace is not None:
        entry["metadata"] = {"cognition_trace": trace}
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def test_cli_prints_pretty_json_by_default(tmp_path, capsys):
    path = tmp_path / "trajectories.jsonl"
    _write_entry(path, {"schema_version": 1, "enabled": True, "route": {"mode": "standard"}})

    exit_code = cognition_trace_report.main([str(path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "\n  \"schema_version\"" in output
    result = json.loads(output)
    assert result["files"] == [str(path)]
    assert result["route"]["modes"] == {"standard": 1}


def test_cli_compact_flag_prints_single_line_json(tmp_path, capsys):
    path = tmp_path / "trajectories.jsonl"
    _write_entry(path, {"schema_version": 1, "enabled": False, "route": {"mode": "disabled"}})

    exit_code = cognition_trace_report.main([str(path), "--compact"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "  " not in output
    result = json.loads(output)
    assert result["route"]["modes"] == {"disabled": 1}


def test_cli_missing_file_does_not_crash(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"

    exit_code = cognition_trace_report.main([str(missing), "--compact"])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["files"] == [str(missing)]
    assert result["errors"]["missing_files"] == 1
