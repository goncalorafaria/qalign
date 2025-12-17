"""
Shared sync/async bridging utilities for HTTP clients.

Goal:
- Provide a stable asyncio event loop per OS thread
- Allow synchronous code paths to call async implementations safely
- Enable aiohttp session reuse via qalign.shared_session (cached per event loop)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_thread_local = threading.local()


def get_or_create_thread_loop() -> asyncio.AbstractEventLoop:
    """Return a stable asyncio event loop for the current thread."""
    loop = getattr(_thread_local, "loop", None)
    if loop is not None and not loop.is_closed():
        return loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _thread_local.loop = loop
    return loop


class ThreadLoopClient:
    """
    Mixin/base class for clients that expose sync wrappers around async methods.

    Important constraints:
    - Do not call sync wrappers from inside a running event loop.
    - The per-thread loop is shared across all instances in that thread.
    """

    def _run_on_thread_loop(self, coro: Coroutine[Any, Any, T]) -> T:
        # If we're already inside an async context, callers should use the async API.
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                f"{self.__class__.__name__} synchronous method called from a running event loop. "
                "Use the async API instead."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

        loop = get_or_create_thread_loop()
        return loop.run_until_complete(coro)

    def __getstate__(self):
        # Deepcopy/pickle safe: no thread-local runtime objects stored on instance.
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)


