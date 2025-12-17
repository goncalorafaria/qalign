"""
Shared aiohttp session management for connection reuse across the application.

This module provides a global session manager that can be initialized at application
startup and shared across all HTTP clients for optimal connection reuse.

Thread-safe: Uses thread-local storage to avoid locks - each thread has its own
session storage, eliminating contention.
"""
import aiohttp
import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SharedSessionManager:
    """
    Manages aiohttp.ClientSession per event loop for reuse across the application.
    
    This allows multiple clients to share the same session within an event loop
    for better connection pooling and resource efficiency.
    """
    
    def __init__(self):
        # Thread-local storage: each thread gets its own session dictionary
        # This avoids the need for locks since threads don't share state
        self._thread_local = threading.local()
        # Connection pool limits - set high to avoid contention
        # Each thread gets its own session, so these limits are per-thread
        # If you have N threads, total connections = N * limit_per_host
        self._default_limit = 10000  # Increased from 2048
        self._default_limit_per_host = 8000  # Increased from 1024 to reduce contention
        self._default_ttl_dns_cache = 300
        self._default_keepalive_timeout = 30
        self._default_timeout = None
    
    def _get_thread_sessions(self) -> dict:
        """Get the thread-local sessions dictionary, creating it if needed."""
        if not hasattr(self._thread_local, 'sessions'):
            self._thread_local.sessions = {}  # event_loop -> (session, connector)
        return self._thread_local.sessions
    
    async def initialize(
        self,
        limit: int = 2048,
        limit_per_host: int = 1024,
        ttl_dns_cache: int = 300,
        keepalive_timeout: int = 30,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ):
        """
        Set default configuration for auto-initialization.
        
        Args:
            limit: Total connection pool size
            limit_per_host: Max connections per host
            ttl_dns_cache: DNS cache TTL in seconds
            keepalive_timeout: Keepalive timeout in seconds
            timeout: Client timeout configuration
        """
        self._default_limit = limit
        self._default_limit_per_host = limit_per_host
        self._default_ttl_dns_cache = ttl_dns_cache
        self._default_keepalive_timeout = keepalive_timeout
        self._default_timeout = timeout
        logger.info("Shared session defaults configured")
    
    async def _ensure_session_for_loop(self, loop: asyncio.AbstractEventLoop) -> aiohttp.ClientSession:
        """
        Ensure a session exists for the given event loop, creating it if needed.
        Thread-safe: Uses thread-local storage, so no locks needed.
        """
        sessions = self._get_thread_sessions()
        
        # Clean up sessions for closed event loops to prevent memory buildup
        # This is important because asyncio.run() creates new loops each time
        closed_loops = []
        for stored_loop, (session, connector) in list(sessions.items()):
            # Check if the stored loop is closed
            if stored_loop.is_closed():
                closed_loops.append(stored_loop)
                logger.debug(
                    f"Cleaning up session for closed event loop {id(stored_loop)} "
                    f"(thread {threading.current_thread().ident})"
                )
                # Try to close the session if it's still open
                if session is not None and not session.closed:
                    try:
                        await session.close()
                    except Exception:
                        pass
                if connector is not None:
                    try:
                        await connector.close()
                    except Exception:
                        pass
        
        # Remove closed loop sessions
        for closed_loop in closed_loops:
            del sessions[closed_loop]
        
        # Check if session exists for this loop
        if loop in sessions:
            session, _ = sessions[loop]
            if session is not None and not session.closed:
                #logger.debug(f"Reusing existing shared session for event loop {id(loop)} (thread {threading.current_thread().ident})")
                return session
            # Session is closed, remove it
            logger.warning(
                f"Session for event loop {id(loop)} was closed, recreating. "
                f"This may indicate an issue with session lifecycle management."
            )
            del sessions[loop]
        
        # Create new session for this event loop
        logger.info(f"Creating new shared session for event loop {id(loop)} (thread {threading.current_thread().ident})")
        connector = aiohttp.TCPConnector(
            limit=self._default_limit,
            limit_per_host=self._default_limit_per_host,
            ttl_dns_cache=self._default_ttl_dns_cache,
            use_dns_cache=True,
            keepalive_timeout=self._default_keepalive_timeout,
            enable_cleanup_closed=True,
            force_close=False,
        )
        
        if self._default_timeout is None:
            timeout = aiohttp.ClientTimeout(
                total=300,
                connect=10,
                sock_read=300
            )
        else:
            timeout = self._default_timeout
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        )
        
        # Store session in thread-local storage
        sessions[loop] = (session, connector)
        logger.info(f"Created and cached shared session for event loop {id(loop)} (thread {threading.current_thread().ident})")
        return session
    
    async def get_session(self) -> Optional[aiohttp.ClientSession]:
        """
        Get the shared session for the current event loop, auto-initializing if needed.
        
        Returns:
            The shared session for the current event loop
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("No event loop available")
            return None
        
        return await self._ensure_session_for_loop(loop)
    
    async def close(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Close the shared session(s) and cleanup resources.
        Thread-safe: Uses thread-local storage, so no locks needed.
        Only closes sessions for the current thread.
        
        Args:
            loop: If provided, close only the session for this loop.
                 If None, close all sessions for the current thread.
        """
        sessions = self._get_thread_sessions()
        
        if loop is not None:
            # Close specific loop's session
            if loop in sessions:
                session, connector = sessions[loop]
                del sessions[loop]
                
                if session is not None and not session.closed:
                    await session.close()
                    await asyncio.sleep(0.25)  # Graceful close
                if connector is not None:
                    await connector.close()
                logger.debug(f"Closed shared session for event loop {id(loop)} (thread {threading.current_thread().ident})")
        else:
            # Close all sessions for this thread
            sessions_to_close = list(sessions.items())
            sessions.clear()
            
            for loop_key, (session, connector) in sessions_to_close:
                if session is not None and not session.closed:
                    await session.close()
                if connector is not None:
                    await connector.close()
            logger.info(f"Closed all shared sessions for thread {threading.current_thread().ident}")
    
    def is_initialized(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
        """
        Check if a session is initialized for the given loop.
        Thread-safe: Uses thread-local storage, so no locks needed.
        Only checks sessions for the current thread.
        
        Args:
            loop: Event loop to check. If None, checks current loop.
        """
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return False
        
        sessions = self._get_thread_sessions()
        if loop not in sessions:
            return False
        
        session, _ = sessions[loop]
        return session is not None and not session.closed


# Global session manager instance
_global_session_manager = SharedSessionManager()


async def initialize_shared_session(
    limit: int = 4096,  # Increased default to reduce contention
    limit_per_host: int = 2048,  # Increased default - each thread gets its own session
    ttl_dns_cache: int = 300,
    keepalive_timeout: int = 30,
    timeout: Optional[aiohttp.ClientTimeout] = None,
):
    """
    Initialize the global shared session.
    
    Call this at application startup for optimal connection reuse.
    """
    await _global_session_manager.initialize(
        limit=limit,
        limit_per_host=limit_per_host,
        ttl_dns_cache=ttl_dns_cache,
        keepalive_timeout=keepalive_timeout,
        timeout=timeout,
    )


async def get_shared_session() -> Optional[aiohttp.ClientSession]:
    """
    Get the shared aiohttp session for the current event loop (convenience function).
    Auto-initializes if not present.
    
    Returns:
        The shared session for the current event loop
    """
    return await _global_session_manager.get_session()


async def close_shared_session(loop: Optional[asyncio.AbstractEventLoop] = None):
    """
    Close the shared session(s).
    
    Args:
        loop: If provided, close only the session for this loop.
             If None, close all sessions.
    """
    await _global_session_manager.close(loop=loop)

