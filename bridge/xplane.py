"""X-Plane 12 local web API client (v3, X-Plane 12.4+).

Used for two things only: scoring (truth logging) and, in single-pilot scenarios,
the actions a copilot may take. Crosscheck's perception never reads from here.
"""
import asyncio
import itertools
import json
import time
from collections.abc import AsyncIterator

import httpx
import websockets

BASE = "http://127.0.0.1:8086/api/v3"
WS = "ws://127.0.0.1:8086/api/v3"


class XPlaneError(RuntimeError):
    pass


class XPlane:
    def __init__(self, base: str = BASE, ws: str = WS):
        self.base, self.ws = base, ws
        self.http = httpx.AsyncClient(base_url=base, timeout=30)
        self._ids: dict[str, int] = {}
        self._cmd_ids: dict[str, int] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, **kw) -> httpx.Response:
        r = await self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise XPlaneError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r

    async def start_flight(self, flight: dict) -> None:
        await self._call("POST", "/flight", json={"data": flight})

    async def dataref_count(self) -> int:
        return (await self._call("GET", "/datarefs/count")).json()["data"]

    async def wait_until_loaded(self, timeout: float = 120) -> int:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if (n := await self.dataref_count()) > 0:
                return n
            await asyncio.sleep(1)
        raise XPlaneError("no datarefs after waiting; is a flight loaded?")

    async def dataref_id(self, name: str) -> int:
        if name not in self._ids:
            r = await self._call("GET", "/datarefs", params={"filter[name]": name, "fields": "id,name"})
            data = r.json()["data"]
            if not data:
                raise XPlaneError(f"unknown dataref {name}")
            self._ids[name] = data[0]["id"]
        return self._ids[name]

    async def list_datarefs(self) -> list[dict]:
        return (await self._call("GET", "/datarefs", params={"fields": "id,name,value_type"})).json()["data"]

    async def get(self, name: str, index: int | None = None):
        params = {} if index is None else {"index": index}
        r = await self._call("GET", f"/datarefs/{await self.dataref_id(name)}/value", params=params)
        data = r.json()["data"]
        # the API may return the whole array even when an index is requested
        if index is not None and isinstance(data, list):
            return data[index] if len(data) > index else data[0]
        return data

    async def set(self, name: str, value, index: int | None = None) -> None:
        params = {} if index is None else {"index": index}
        await self._call("PATCH", f"/datarefs/{await self.dataref_id(name)}/value",
                         params=params, json={"data": value})

    async def command(self, name: str, duration: float = 0) -> None:
        if name not in self._cmd_ids:
            r = await self._call("GET", "/commands", params={"filter[name]": name, "fields": "id,name"})
            data = r.json()["data"]
            if not data:
                raise XPlaneError(f"unknown command {name}")
            self._cmd_ids[name] = data[0]["id"]
        await self._call("POST", f"/command/{self._cmd_ids[name]}/activate", json={"duration": duration})

    async def subscribe(self, names: list[str]) -> AsyncIterator[tuple[float, dict[str, object]]]:
        """Yield (monotonic time, {name: value}) for every update; later updates carry only changes."""
        ids = {await self.dataref_id(n): n for n in names}
        req = itertools.count(1)
        async with websockets.connect(self.ws, max_size=None) as ws:
            await ws.send(json.dumps({"req_id": next(req), "type": "dataref_subscribe_values",
                                      "params": {"datarefs": [{"id": i} for i in ids]}}))
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "result" and not msg.get("success", True):
                    raise XPlaneError(f"subscribe failed: {msg}")
                if msg.get("type") == "dataref_update_values":
                    yield time.monotonic(), {ids[int(k)]: v for k, v in msg["data"].items() if int(k) in ids}
