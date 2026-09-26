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
load ``~/.hermes/config.yaml`` themselves). Pure: no I/O for
``get_cognition_config``; ``load_cognition_config_from_home`` delegates the
one bounded read to ``hermes_cli.config.load_config_readonly()`` (the
canonical, guard-enforced loader — see v0.20's
``tests/hermes_cli/test_config_read_guard.py``) rather than reading the
YAML file itself.

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
    "interaction_stance",
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

    Routes through ``hermes_cli.config.load_config_readonly()`` scoped to
    ``hermes_home`` via ``set_hermes_home_override`` (the same seam
    ``_profile_runtime_scope`` / ``profiles.py`` / ``kanban_db.py`` use to
    read another home's config) rather than a bare ``yaml.safe_load`` of the
    file. A bare read would silently skip the managed-scope overlay,
    ``${ENV_VAR}`` expansion, profile-aware pathing, and root-model
    normalization that the canonical loader applies — see
    ``tests/hermes_cli/test_config_read_guard.py``.

    Returns ``{}`` on any failure (home unreadable, YAML parse error,
    cognition block missing or malformed). Note that as of v0.20 the
    ``cognition`` block ships as part of ``DEFAULT_CONFIG`` (see
    ``hermes_cli/config_defaults.py``), so a ``hermes_home`` with no
    ``config.yaml`` at all now returns the compiled-in cognition defaults
    (normalized) rather than ``{}`` — this matches how every other config
    section behaves once loaded through the canonical loader.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(hermes_home))
        try:
            raw = load_config_readonly()
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        logger.debug("cognition config read failed (non-fatal): %s", exc)
        return {}
    return get_cognition_config(raw)
