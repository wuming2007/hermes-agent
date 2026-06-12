#!/usr/bin/env python3
"""Offline cognition trace report CLI.

Reads trajectory JSONL files produced by PR8 and prints a deterministic JSON
summary of `metadata.cognition_trace` without touching runtime behavior.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agent.cognition_trace_report import analyze_cognition_trace_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an offline cognition trace report from trajectory JSONL files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Trajectory JSONL file(s) to analyze.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact single-line JSON instead of pretty JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = analyze_cognition_trace_jsonl(args.paths)
    if args.compact:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
