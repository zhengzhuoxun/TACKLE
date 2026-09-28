"""
Generic deduplication utility used across pipeline stages.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")


def deduplicate(items: list[T], key: Callable[[T], K]) -> list[T]:
    """Remove duplicates from a list, preserving first-occurrence order.

    Args:
        items: The list to deduplicate.
        key: A callable that extracts a hashable key from each item.

    Returns:
        A new list with duplicates removed.
    """
    seen: set[K] = set()
    result: list[T] = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        result.append(item)
    return result
