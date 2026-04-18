"""Shared loader / normalizer for the ``cognition`` config block (PR4).

Before PR4 each production entry point (``cli.py``, ``gateway/run.py``,
``cron/scheduler.py``, and ``AIAgent.__init__``) parsed the cognition
block its own way. That risked the four entry points drifting on:

- handling of malformed sub-blocks (a stray ``fast_mode: "broken"`` would
  crash one entry point but be tolerated by another),
- env-expansion / future migration semantics,
- default filling vs strict pass-through.

This module is the single source of truth. All entry points must call
``get_cognition_config(...)`` (when they already have the parsed config
dict) or ``load_cognition_config_from_home(home)`` (when they need to
read raw YAML from ``~/.hermes/config.yaml``). Pure: no I/O for
``get_cognition_config``, only one well-bounded YAML read for
``load_cognition_config_from_home``.

Normalization rules — kept narrow on purpose so PR4 stays a cleanup
rather than a behavior change:

- ``None`` / non-dict input → ``{}``
- missing ``cognition`` key → ``{}``
- ``cognition`` value that is not a dict → ``{}``
- valid dict → shallow-copied; recognized sub-blocks (``fast_mode``,
  ``deep_mode_triggers``, ``consistency_guard``) that are present but
  not dicts get coerced to ``{}`` so downstream ``.get()`` calls cannot
  raise.

Defaults are NOT injected here — callers that need defaults read via
``.get(key, default)`` against the normalized dict. This keeps the
contract explicit ("user provided X, didn't provide Y") rather than
hiding it under default-merging.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_RECOGNIZED_SUB_BLOCKS: tuple[str, ...] = (
    "fast_mode",
    "deep_mode_triggers",
    "consistency_guard",
)


def get_cognition_config(config: Any) -> dict:
    """Return the normalized cognition block from a full config dict.

    See module docstring for the normalization rules.
    """
    if not isinstance(config, dict):
        return {}
    raw = config.get("cognition")
    if not isinstance(raw, dict):
        return {}
    # Deep copy so callers can mutate the result without disturbing the
    # source config (entry points sometimes hand the dict around to other
    # subsystems that assume immutability).
    out = copy.deepcopy(raw)
    for sub in _RECOGNIZED_SUB_BLOCKS:
        if sub in out and not isinstance(out[sub], dict):
            out[sub] = {}
    return out


def load_cognition_config_from_home(hermes_home: Path) -> dict:
    """Read ``<hermes_home>/config.yaml`` and return the normalized cognition block.

    Returns ``{}`` on any failure (file missing, YAML parse error,
    cognition block missing or malformed). This matches the
    pre-PR4 ``_load_cognition_config`` shape gateway used to ship.
    """
    try:
        cfg_path = Path(hermes_home) / "config.yaml"
        if not cfg_path.exists():
            return {}
        import yaml

        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.debug("cognition config read failed (non-fatal): %s", exc)
        return {}
    return get_cognition_config(raw)
