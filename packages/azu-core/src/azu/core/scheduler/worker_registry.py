"""
WorkerRegistry
==============
Redis-backed static registry for worker endpoint records.

Separates two concerns previously conflated inside MoEScheduler.workers:

  Endpoint knowledge  (this module — durable, survives restarts):
    • worker_id, type (persistent | serverless), invoke URL
    • hardware specs: VRAM, capabilities, platform
    • payment address
    • static P2P URL for persistent workers

  Session state  (MoEScheduler.workers — ephemeral):
    • open WebSocket handle
    • runtime VRAM accounting
    • heartbeat timestamp
    • cached layer set

The registry is the authoritative source for serverless workers, which have
no persistent connection.  It also lets the scheduler rebuild self.workers
from durable state on restart, so existing persistent workers re-appear as
soon as they reconnect.

Redis key layout
----------------
  registry:worker:<worker_id>   JSON blob (WorkerEndpoint)
  registry:workers              SET  of worker_ids (membership index)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

import redis.asyncio as aioredis


# ── key constants ─────────────────────────────────────────────────────────────
_KEY_PREFIX = "registry:worker:"
_INDEX_KEY  = "registry:workers"


# ── domain types ──────────────────────────────────────────────────────────────

class WorkerType(str, Enum):
    PERSISTENT = "persistent"   # maintains long-lived WebSocket to scheduler
    SERVERLESS = "serverless"   # invoked on-demand via HTTP; no persistent connection


@dataclass
class WorkerEndpoint:
    """
    Static record written at deploy/registration time.

    Must NOT carry ephemeral runtime fields (WebSocket handles, heartbeat
    timestamps, per-job VRAM ledger, etc.).  Those live in WorkerState.
    """
    worker_id:       str
    worker_type:     WorkerType
    endpoint_url:    str            # ws:// for persistent; https:// invoke URL for serverless
    vram_mb:         int
    capabilities:    List[str]      = field(default_factory=lambda: ["dense", "moe_router", "moe_expert"])
    platform:        str            = "self_hosted"
    payment_address: Optional[str]  = None
    # Known static P2P URL.  Set for persistent workers on registration.
    # For serverless workers this starts None and is patched by update_p2p_url()
    # once the worker calls POST /worker/ready after cold-start.
    p2p_url:         Optional[str]  = None
    registered_at:   float          = field(default_factory=time.time)
    updated_at:      float          = field(default_factory=time.time)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["worker_type"] = self.worker_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerEndpoint":
        d = dict(d)
        d["worker_type"] = WorkerType(d["worker_type"])
        return cls(**d)


# ── registry ──────────────────────────────────────────────────────────────────

class WorkerRegistry:
    """
    Thin async CRUD layer for WorkerEndpoint records in Redis.

    Only this class writes to registry:* keys.  The scheduler reads records
    through this class; it must not manipulate those keys directly.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._r = redis_client

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _key(worker_id: str) -> str:
        return f"{_KEY_PREFIX}{worker_id}"

    # ── write ─────────────────────────────────────────────────────────────────

    async def register(self, endpoint: WorkerEndpoint) -> None:
        """
        Upsert a worker record.

        Safe to call on every reconnect: subsequent calls update updated_at
        without losing accumulated p2p_url or payment_address.
        """
        endpoint.updated_at = time.time()
        raw = json.dumps(endpoint.to_dict())
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.set(self._key(endpoint.worker_id), raw)
            pipe.sadd(_INDEX_KEY, endpoint.worker_id)
            await pipe.execute()
        print(f"📋 [Registry] Registered {endpoint.worker_id[:16]} "
              f"type={endpoint.worker_type.value} vram={endpoint.vram_mb}MB")

    async def deregister(self, worker_id: str) -> bool:
        """
        Remove a worker record permanently.

        Returns True if the record existed.  Deregistering a worker that is
        still actively connected is operator error; the caller is responsible
        for also removing it from MoEScheduler.workers.
        """
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.delete(self._key(worker_id))
            pipe.srem(_INDEX_KEY, worker_id)
            results = await pipe.execute()
        existed = bool(results[0])
        if existed:
            print(f"📋 [Registry] Deregistered {worker_id[:16]}")
        return existed

    async def update_p2p_url(self, worker_id: str, p2p_url: str) -> bool:
        """
        Patch the p2p_url for a worker that has just become reachable.

        Called by POST /worker/ready when a serverless worker finishes
        cold-starting and knows its ephemeral inbound URL.

        Returns False if the worker_id is not in the registry.
        """
        ep = await self.get(worker_id)
        if ep is None:
            return False
        ep.p2p_url    = p2p_url
        ep.updated_at = time.time()
        await self._r.set(self._key(worker_id), json.dumps(ep.to_dict()))
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    async def get(self, worker_id: str) -> Optional[WorkerEndpoint]:
        raw = await self._r.get(self._key(worker_id))
        if raw is None:
            return None
        return WorkerEndpoint.from_dict(json.loads(raw))

    async def list_all(self) -> List[WorkerEndpoint]:
        """Return all registered workers regardless of type or connectivity."""
        worker_ids = await self._r.smembers(_INDEX_KEY)
        if not worker_ids:
            return []
        keys   = [self._key(wid) for wid in worker_ids]
        values = await self._r.mget(*keys)
        out: List[WorkerEndpoint] = []
        for raw in values:
            if raw:
                try:
                    out.append(WorkerEndpoint.from_dict(json.loads(raw)))
                except Exception:
                    # Corrupt record — skip silently rather than crashing the scheduler.
                    pass
        return out

    async def list_by_type(self, worker_type: WorkerType) -> List[WorkerEndpoint]:
        """Convenience filter: return only persistent or only serverless workers."""
        return [ep for ep in await self.list_all() if ep.worker_type == worker_type]

    async def exists(self, worker_id: str) -> bool:
        return bool(await self._r.exists(self._key(worker_id)))
