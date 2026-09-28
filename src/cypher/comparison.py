"""Value normalization used when loading the instance KG into Kuzu."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def to_number(value: Any) -> float | None:
    """Best-effort numeric coercion (bools are not numbers)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_iso_date(value: Any) -> date | None:
    """Best-effort ISO date coercion."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None
