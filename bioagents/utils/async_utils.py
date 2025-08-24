"""
Async utilities for Streamlit applications.

This module provides utilities to handle async operations properly in Streamlit,
avoiding common issues like "Event loop is closed" exceptions.

Part of the bioagents package utilities.
"""

import asyncio
import concurrent.futures
import sys
from typing import Any, Callable, Coroutine


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine in a Streamlit-compatible way.

    This function handles the event loop properly to avoid "Event loop is closed"
    exceptions that commonly occur in Streamlit applications.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        Exception: Any exception raised by the coroutine
    """

    def _run_in_new_loop():
        """Run the coroutine in a new event loop."""
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    try:
        # Try to use existing event loop
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop, so use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result()
    except RuntimeError:
        # No event loop exists or it's not running, create one
        return _run_in_new_loop()


def create_async_wrapper(async_func: Callable[..., Coroutine[Any, Any, Any]]):
    """
    Create a synchronous wrapper for an async function.

    This decorator or factory function creates a synchronous wrapper that
    properly handles event loops in Streamlit applications.

    Args:
        async_func: The async function to wrap

    Returns:
        A synchronous function that calls the async function
    """

    def sync_wrapper(*args, **kwargs):
        """Synchronous wrapper that calls the async function."""
        return run_async(async_func(*args, **kwargs))

    return sync_wrapper
