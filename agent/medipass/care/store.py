"""Where doctor requests are kept.

Upstash Redis when UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN (or Vercel's KV_REST_API_*) are set, so the queue survives
serverless hosting. Otherwise this process's memory, which is fine locally and on an always-on server.
"""

import json
import os
from collections.abc import Mapping
from typing import Any

import httpx

TTL_SECONDS = 24 * 3600


class Store:
    def __init__(self, env: Mapping[str, str] | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        env = os.environ if env is None else env
        self._url = (env.get("UPSTASH_REDIS_REST_URL") or env.get("KV_REST_API_URL") or "").rstrip("/")
        self._token = env.get("UPSTASH_REDIS_REST_TOKEN") or env.get("KV_REST_API_TOKEN") or ""
        self._transport = transport
        self._local: dict[str, dict[str, Any]] = {}
        self.persistent = bool(self._url and self._token)

    async def _redis(self, command: list[Any]) -> Any:
        async with httpx.AsyncClient(timeout=5, transport=self._transport) as client:
            res = await client.post(self._url, json=command, headers={"authorization": f"Bearer {self._token}"})
        res.raise_for_status()
        return res.json().get("result")

    async def put(self, record: dict[str, Any]) -> None:
        if not self.persistent:
            self._local[record["id"]] = record
            return
        await self._redis(["SET", f"medipass:req:{record['id']}", json.dumps(record), "EX", TTL_SECONDS])
        await self._redis(["SADD", "medipass:reqs", record["id"]])

    async def get(self, request_id: str) -> dict[str, Any] | None:
        if not self.persistent:
            return self._local.get(request_id)
        raw = await self._redis(["GET", f"medipass:req:{request_id}"])
        return json.loads(raw) if raw else None

    async def all(self) -> list[dict[str, Any]]:
        if not self.persistent:
            records = list(self._local.values())
        else:
            ids = await self._redis(["SMEMBERS", "medipass:reqs"]) or []
            records = [r for r in [await self.get(i) for i in ids] if r]
        return sorted(records, key=lambda r: r["createdAt"], reverse=True)


store = Store()
