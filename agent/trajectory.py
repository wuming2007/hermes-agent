"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def build_trajectory_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-friendly deep copy of optional trajectory metadata.

    ``save_trajectory`` historically wrote only the conversation, model, and
    completion flag.  PR8 keeps that old shape when metadata is absent, while
    allowing downstream cognition trace consumers to receive stable top-level
    metadata. Non-JSON values are stringified instead of failing the save path.
    """
    if not isinstance(metadata, Mapping) or not metadata:
        return None

    return json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None,
                    metadata: Mapping[str, Any] | None = None):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
        metadata: Optional JSON-friendly entry metadata. Omitted when absent.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }
    normalized_metadata = build_trajectory_metadata(metadata)
    if normalized_metadata is not None:
        entry["metadata"] = normalized_metadata

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
