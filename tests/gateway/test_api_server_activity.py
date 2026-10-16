"""``/v1/activity`` + ``/v1/activity/events`` (read-only CLI activity feed)."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_activity
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.cli_activity import CliActivityPublisher

KEY = "sk-activity-test-0123456789"


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/activity") or path == "/v1/capabilities":
            app.router.add_route(method, path, handler)
    return app


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(api_server_activity, "POLL_INTERVAL_SECONDS", 0.02)
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))


def _auth() -> dict:
    return {"Authorization": f"Bearer {KEY}"}


async def _next_event(resp) -> dict:
    while True:
        line = (await asyncio.wait_for(resp.content.readline(), 5)).decode()
        if line.startswith("data: "):
            return json.loads(line[6:])


@pytest.mark.asyncio
async def test_activity_requires_the_api_key(adapter) -> None:
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await cli.get("/v1/activity")).status == 401
        assert (await cli.get("/v1/activity/events")).status == 401


@pytest.mark.asyncio
async def test_capabilities_advertise_the_feed(adapter) -> None:
    async with TestClient(TestServer(_app(adapter))) as cli:
        caps = await (await cli.get("/v1/capabilities", headers=_auth())).json()
    assert caps["features"]["activity_feed"]["supported"] is True
    assert caps["endpoints"]["activity_events"] == {"method": "GET", "path": "/v1/activity/events"}


@pytest.mark.asyncio
async def test_snapshot_then_updates_then_removal(adapter) -> None:
    from hermes_constants import get_hermes_home
    pub = CliActivityPublisher(get_hermes_home(), profile="default")
    pub.turn_started("sess-a")
    async with TestClient(TestServer(_app(adapter))) as cli:
        listed = await (await cli.get("/v1/activity", headers=_auth())).json()
        assert [a["state"] for a in listed["activities"]] == ["thinking"]

        resp = await cli.get("/v1/activity/events", headers=_auth())
        snap = await _next_event(resp)
        assert snap["event"] == "activity.snapshot"
        assert [(a["session_id"], a["state"]) for a in snap["activities"]] == [("sess-a", "thinking")]

        pub.tool_started("terminal")
        ev = await _next_event(resp)
        assert ev["event"] == "activity.updated" and ev["activity"]["state"] == "working"
        assert ev["activity"]["seq"] > snap["activities"][0]["seq"]

        pub.turn_finished({"completed": True})
        ev = await _next_event(resp)
        assert ev["activity"]["state"] == "completed"

        pub.path.unlink()  # retention over / swept
        ev = await _next_event(resp)
        assert ev["event"] == "activity.removed" and ev["id"] == snap["activities"][0]["id"]
        resp.close()
    await asyncio.sleep(0.05)
    hub = next(iter(adapter._activity_hubs.values()))
    assert hub._task is None and not hub._subscribers  # polling stops with the last client
