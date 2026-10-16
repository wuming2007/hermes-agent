"""``/v1/activity`` routes: read-only view of this profile's CLI turn activity.

The CLI processes publish their own lifecycle records (``hermes_cli.cli_activity``); this
module only reads them. ``GET /v1/activity`` returns the current list; ``GET
/v1/activity/events`` is SSE: one ``activity.snapshot`` on connect, then
``activity.updated`` / ``activity.removed`` as records change. Each connection gets the full
snapshot first, so a client that reconnects recovers the real state instead of guessing.
No route here can answer an approval, stop a CLI turn or read conversation content.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 0.2


def _event(name: str, **fields: Any) -> dict[str, Any]:
    return {"event": name, "timestamp": time.time(), **fields}


class ActivityFeedHub:
    """Polls one HERMES_HOME's activity records while at least one client is subscribed."""

    def __init__(self, hermes_home: Path, *, interval: Optional[float] = None) -> None:
        self.hermes_home = Path(hermes_home)
        self.interval = POLL_INTERVAL_SECONDS if interval is None else interval
        self._subscribers: set[asyncio.Queue] = set()
        self._current: dict[str, dict[str, Any]] = {}
        self._task: Optional[asyncio.Task] = None

    async def read(self) -> list[dict[str, Any]]:
        from hermes_cli.cli_activity import read_activities
        return await asyncio.to_thread(read_activities, self.hermes_home)

    async def subscribe(self) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        activities = await self.read()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        if self._task is None or self._task.done():
            self._current = {a["id"]: a for a in activities}
            self._task = asyncio.get_running_loop().create_task(self._run())
        return queue, activities

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
        if not self._subscribers and self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while self._subscribers:
            await asyncio.sleep(self.interval)
            try:
                activities = await self.read()
            except Exception:
                logger.debug("activity read failed", exc_info=True)
                continue
            self._publish(activities)

    def _publish(self, activities: list[dict[str, Any]]) -> None:
        latest = {a["id"]: a for a in activities}
        events = [_event("activity.updated", activity=a) for k, a in latest.items() if self._current.get(k) != a]
        events += [_event("activity.removed", id=k, activity=a) for k, a in self._current.items() if k not in latest]
        self._current = latest
        for event in events:
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # A wedged client: drop it rather than grow memory; it reconnects to a snapshot.
                    self._subscribers.discard(queue)
                    with suppress(asyncio.QueueFull):
                        queue.put_nowait(None)


def _hub(adapter: Any) -> ActivityFeedHub:
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    hubs = adapter.__dict__.setdefault("_activity_hubs", {})
    hub = hubs.get(str(home))
    if hub is None:
        hub = hubs[str(home)] = ActivityFeedHub(home)
    return hub


def _profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return ""


async def _handle_activity(self, request, *, _api_server):
    from aiohttp import web
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    activities = await _hub(self).read()
    return web.json_response({"object": "hermes.activity", "profile": _profile(), "schema": 1,
                              "activities": activities})


async def _handle_activity_events(self, request, *, _api_server):
    from aiohttp import web
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    hub = _hub(self)
    queue, activities = await hub.subscribe()
    response = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        await response.prepare(request)
        await response.write(_api_server._sse_frame(
            _event("activity.snapshot", profile=_profile(), activities=activities)))
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
                continue
            if event is None:
                break
            await response.write(_api_server._sse_frame(event))
    except (ConnectionResetError, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.debug("[api_server] activity SSE ended: %s", exc)
    finally:
        hub.unsubscribe(queue)
    return response


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [("GET", "/v1/activity", self._handle_activity),
            ("GET", "/v1/activity/events", self._handle_activity_events)]


CAPABILITY = {"supported": True, "schema": 1, "surfaces": ["cli"],
              "states": ["thinking", "working", "talking", "waiting", "completed", "failed", "cancelled", "lost"]}
