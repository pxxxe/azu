"""
WorkerDriver
============
Transport abstraction for scheduler → worker control-plane messages.

The MoEScheduler uses drivers to deliver JSON control payloads to workers
without knowing whether the worker is a persistent WebSocket client or a
serverless HTTP endpoint.  Tensor data never flows through either driver —
tensors move directly between workers over P2P.

Two concrete implementations
-----------------------------
  PersistentDriver   — pushes JSON over the worker's open WebSocket
  ServerlessDriver   — POSTs JSON to the worker's HTTP invoke endpoint

Selection
---------
  Call get_driver_for_worker(worker_state) and it returns the right driver
  based on worker_state.specs["worker_type"].  No other code needs to branch
  on worker type for dispatch.

Session lifecycle
-----------------
  ServerlessDriver maintains a single aiohttp.ClientSession for the process
  lifetime.  It is lazily created on first use and is safe to share across
  concurrent dispatch calls.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import aiohttp

if TYPE_CHECKING:
    # Avoid circular import; WorkerState is defined in scheduler/main.py
    from azu.core.scheduler.main import WorkerState


# ── abstract base ─────────────────────────────────────────────────────────────

class WorkerDriver(ABC):
    """One method.  One contract.  No tensor data."""

    @abstractmethod
    async def send_control(self, worker: "WorkerState", payload: dict) -> None:
        """
        Deliver a control-plane JSON message to a worker.

        Raises RuntimeError on unrecoverable transport failure so the caller
        can decide whether to retry, re-queue the job, or drop the worker.
        """
        ...


# ── persistent (WebSocket) ────────────────────────────────────────────────────

class PersistentDriver(WorkerDriver):
    """
    Delivers control messages over the worker's live WebSocket connection.

    ws.send_json is already async-safe in Starlette/FastAPI; no extra
    locking is required.
    """

    async def send_control(self, worker: "WorkerState", payload: dict) -> None:
        if worker.ws is None:
            raise RuntimeError(
                f"PersistentDriver: worker {worker.pubkey!r} has no active WebSocket. "
                "Was it registered as persistent but reconnected before the socket was set?"
            )
        await worker.ws.send_json(payload)


# ── serverless (HTTP) ─────────────────────────────────────────────────────────

class ServerlessDriver(WorkerDriver):
    """
    Delivers control messages via HTTP POST to the worker's invoke endpoint.

    The invoke URL is stored in worker.specs["endpoint_url"].  It is written
    into the registry at deploy time (e.g. a RunPod Serverless run URL).

    The worker-side HTTP handler must accept the same JSON payload shapes as
    the existing WebSocket receive loop:
      JOB_START, EXECUTE_DENSE, EXECUTE_ROUTER, EXECUTE_EXPERT

    A single shared aiohttp.ClientSession is lazily created on first use and
    reused for all subsequent calls.  It is protected by an asyncio.Lock so
    creation is safe under concurrent startup.
    """

    _TIMEOUT = aiohttp.ClientTimeout(total=30)

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._init_lock = asyncio.Lock()

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._init_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=self._TIMEOUT,
                        headers={"Content-Type": "application/json"},
                    )
        return self._session

    async def send_control(self, worker: "WorkerState", payload: dict) -> None:
        endpoint_url = worker.specs.get("endpoint_url")
        if not endpoint_url:
            raise RuntimeError(
                f"ServerlessDriver: worker {worker.pubkey!r} is missing endpoint_url "
                "in specs.  Was it registered via POST /workers with a valid endpoint_url?"
            )
        session = await self._session_get()
        async with session.post(endpoint_url, json=payload) as resp:
            if resp.status >= 300:
                body = await resp.text()
                raise RuntimeError(
                    f"ServerlessDriver: POST to {endpoint_url} returned "
                    f"HTTP {resp.status}: {body[:300]}"
                )

    async def close(self) -> None:
        """Graceful shutdown.  Call from the FastAPI shutdown event if needed."""
        async with self._init_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None


# ── module-level singletons ───────────────────────────────────────────────────
# One instance per transport type.  Instantiated lazily; no I/O at import time.

_persistent_driver  = PersistentDriver()
_serverless_driver: Optional[ServerlessDriver] = None
_serverless_init_lock = asyncio.Lock()


def get_persistent_driver() -> PersistentDriver:
    return _persistent_driver


async def get_serverless_driver() -> ServerlessDriver:
    global _serverless_driver
    if _serverless_driver is None:
        async with _serverless_init_lock:
            if _serverless_driver is None:
                _serverless_driver = ServerlessDriver()
    return _serverless_driver


def get_driver_for_worker(worker: "WorkerState") -> WorkerDriver:
    """
    Return the correct driver for a WorkerState without awaiting.

    ServerlessDriver creates its HTTP session lazily on the first
    send_control() call, so it is safe to return here synchronously.
    """
    global _serverless_driver
    if worker.specs.get("worker_type") == "serverless":
        if _serverless_driver is None:
            _serverless_driver = ServerlessDriver()
        return _serverless_driver
    return _persistent_driver
