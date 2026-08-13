"""
Utils — Performance timing decorator.

Measures and logs execution time of functions. Useful for profiling
each pipeline stage (detection, embedding, etc.) without modifying
business logic.

Usage:
    from utils.timing import timed

    @timed
    def detect_faces(frame):
        ...
"""

from __future__ import annotations

import functools
import time
from typing import Callable, Optional

from utils.logger import get_logger

_logger = get_logger("tracify.timing")


def timed(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator that logs execution time of the wrapped function.

    Can be used with or without arguments:
        @timed
        def foo(): ...

        @timed(name="custom_label")
        def bar(): ...

    Args:
        func: The function to wrap (when used without parentheses).
        name: Optional custom name for the log entry.
    """

    def decorator(fn: Callable) -> Callable:
        label = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                _logger.debug(
                    f"{label} completed in {elapsed_ms:.1f}ms",
                    extra={"function": label, "elapsed_ms": round(elapsed_ms, 1)},
                )
                return result
            except Exception:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _logger.error(
                    f"{label} failed after {elapsed_ms:.1f}ms",
                    extra={"function": label, "elapsed_ms": round(elapsed_ms, 1)},
                    exc_info=True,
                )
                raise

        return wrapper

    # Support both @timed and @timed(name="...")
    if func is not None:
        return decorator(func)
    return decorator
