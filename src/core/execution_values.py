"""
Deterministic value helpers for formatting Cypher execution results.
"""

from __future__ import annotations

from typing import Any


def dedupe_rows(rows: list[list[Any]]) -> list[list[Any]]:
    """Remove duplicate projected rows."""
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[Any]] = []
    for row in rows:
        key = tuple(stringify(value) for value in row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def stringify(value: Any) -> str:
    """Normalize scalar values for output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_final_answer(value: Any) -> str:
    """Render the final execution value into the project answer format."""
    if value is None:
        return "(no result)"
    if isinstance(value, list):
        if not value:
            return "(no result)"
        if all(isinstance(row, list) for row in value):
            rows = value
            if len(rows) == 1:
                return ", ".join(stringify(cell) for cell in rows[0])
            if all(len(row) == 1 for row in rows):
                return ", ".join(stringify(row[0]) for row in rows)
            return " | ".join(", ".join(stringify(cell) for cell in row) for row in rows)
        return ", ".join(stringify(item) for item in value)
    return stringify(value)
