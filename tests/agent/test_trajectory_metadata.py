import json

from agent.trajectory import build_trajectory_metadata, save_trajectory


class NotJsonSerializable:
    def __str__(self) -> str:
        return "not-json-serializable"


def _read_single_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_build_trajectory_metadata_returns_none_without_metadata():
    assert build_trajectory_metadata(None) is None
    assert build_trajectory_metadata({}) is None


def test_save_trajectory_omits_metadata_key_when_not_provided(tmp_path):
    filename = tmp_path / "trajectory.jsonl"

    save_trajectory(
        [{"from": "human", "value": "hello"}],
        "test-model",
        True,
        filename=str(filename),
    )

    entry = _read_single_jsonl(filename)
    assert entry["conversations"] == [{"from": "human", "value": "hello"}]
    assert entry["model"] == "test-model"
    assert entry["completed"] is True
    assert "metadata" not in entry


def test_save_trajectory_exports_cognition_trace_metadata(tmp_path):
    filename = tmp_path / "trajectory.jsonl"
    trace = {
        "schema_version": 1,
        "enabled": True,
        "route": {"mode": "standard"},
        "uncertainty": {"present": False},
        "verification": {"ladder_enabled": True},
    }

    save_trajectory(
        [{"from": "gpt", "value": "hi"}],
        "test-model",
        True,
        filename=str(filename),
        metadata={"cognition_trace": trace},
    )

    entry = _read_single_jsonl(filename)
    assert entry["metadata"] == {"cognition_trace": trace}


def test_build_trajectory_metadata_does_not_mutate_input():
    trace = {"schema_version": 1, "route": {"mode": "fast"}}
    metadata = {"cognition_trace": trace}

    normalized = build_trajectory_metadata(metadata)

    assert normalized == metadata
    assert normalized is not metadata
    assert normalized["cognition_trace"] is not trace
    assert metadata == {"cognition_trace": {"schema_version": 1, "route": {"mode": "fast"}}}


def test_build_trajectory_metadata_coerces_non_serializable_values_to_strings():
    normalized = build_trajectory_metadata(
        {
            "cognition_trace": {
                "schema_version": 1,
                "problem": NotJsonSerializable(),
            }
        }
    )

    assert normalized == {
        "cognition_trace": {
            "schema_version": 1,
            "problem": "not-json-serializable",
        }
    }
