"""Behavior contract: only the user's own words in a live turn count as the owner speaking.

Owner turns are what may grant authority, create tasks or cancel them, so anything the runtime
injects in a user-role row must not pass.
"""

import pytest

from plugins.cognition_wm.model import SourceRef
from plugins.cognition_wm.provenance import is_owner_turn


@pytest.mark.parametrize(
    "message, platform, delegated, owner",
    [
        pytest.param({"role": "user", "content": "只要幫我寫 prompt，不要改 repo"}, "cli", False, True, id="user-words"),
        pytest.param({"role": "user", "content": [{"type": "text", "text": "看這張圖"}]}, "telegram", False, True,
                     id="multimodal-user-words"),
        pytest.param({"role": "user", "content": "[IMPORTANT: Background process p1 completed normally (exit code 0).\n"
                                                 "Command: ./deploy.sh"}, "cli", False, False, id="job-notice"),
        pytest.param({"role": "user", "content": "[ASYNC DELEGATION COMPLETE — d1]\nresult"}, "cli", False, False,
                     id="delegation-notice"),
        pytest.param({"role": "user", "content": "[System note: The following is recalled memory context"}, "cli", False,
                     False, id="system-note"),
        pytest.param({"role": "user", "content": "[Background process p1 heartbeat #2 — still running"}, "cli", False,
                     False, id="heartbeat"),
        pytest.param({"role": "user", "content": "keep going", "_verification_stop_synthetic": True}, "cli", False, False,
                     id="runtime-nudge"),
        pytest.param({"role": "user", "content": "ship it"}, "cron", False, False, id="cron-run"),
        pytest.param({"role": "user", "content": "ship it"}, "cli", True, False, id="delegated-child"),
        pytest.param({"role": "assistant", "content": "ship it"}, "cli", False, False, id="not-a-user-row"),
    ],
)
def test_owner_turns_are_the_users_own_words(message, platform, delegated, owner):
    assert is_owner_turn(message, platform=platform, delegated=delegated) is owner


def test_source_refs_round_trip_and_reject_unknown_authors():
    ref = SourceRef(type="tool_result", session_id="s", turn_id="t", ref="c1", at=1.0, authored_by="tool", run_id="r")

    assert SourceRef.from_json(ref.to_json()) == ref
    with pytest.raises(ValueError):
        SourceRef(type="user_message", session_id="s", turn_id="t", ref="m", at=1.0, authored_by="admin", run_id="r")
