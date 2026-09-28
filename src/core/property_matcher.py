"""Deterministic property alignment for Step 2 (class merging).

When two clusters are merged, their properties can carry different raw column
names (e.g. ``stu_nr`` vs ``studentID``) and different canonical names. This
module scores and aligns an incoming property against an entity's existing
properties so the merger can collapse them into one canonical property and
union their provenance — no LLM call required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.strings import (
    description_similarity,
    name_similarity,
    snake_case,
    strip_table_prefixes,
)

if TYPE_CHECKING:
    from src.core.cluster_merger import EntityProperty


def _value_overlap_score(
    left: list[object],
    right: list[object],
) -> float:
    """Jaccard over sample values, compared by string representation."""
    if not left or not right:
        return 0.0
    a = {repr(v) for v in left}
    b = {repr(v) for v in right}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def property_similarity(
    incoming: "EntityProperty",
    existing: "EntityProperty",
) -> float:
    """Score how likely two properties are the same concept (0..1)."""
    name_score = name_similarity(incoming.name, existing.name)

    # Best raw-column-name match across the two source column sets.
    col_score = 0.0
    if incoming.source_columns and existing.source_columns:
        best = 0.0
        for left_col in incoming.source_columns:
            for right_col in existing.source_columns:
                score = name_similarity(left_col, right_col)
                if score > best:
                    best = score
        col_score = best

    desc_score = description_similarity(
        incoming.description,
        existing.description,
    )

    value_score = _value_overlap_score(
        list(incoming.sample_values),
        list(existing.sample_values),
    )

    return round(
        0.35 * name_score
        + 0.30 * col_score
        + 0.20 * desc_score
        + 0.15 * value_score,
        4,
    )


def match_property(
    incoming: "EntityProperty",
    existing_properties: list["EntityProperty"],
    threshold: float = 0.5,
) -> "EntityProperty | None":
    """Return the existing property best matching ``incoming``, if any."""
    best: "EntityProperty | None" = None
    best_score = 0.0
    for existing in existing_properties:
        score = property_similarity(incoming, existing)
        if score > best_score:
            best_score = score
            best = existing
    if best is not None and best_score >= threshold:
        return best
    return None


def canonicalize_property_name(name: str, table_names: list[str] | None = None) -> str:
    """Normalise a property name to a transparent snake_case form.

    Strips leading table-name tokens so ``dept_department_id`` becomes
    ``department_id``.
    """
    cleaned = snake_case(name)
    if table_names:
        cleaned = strip_table_prefixes(cleaned, table_names)
    return cleaned or snake_case(name)
