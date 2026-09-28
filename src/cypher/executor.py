"""Deterministic Cypher executor.

Runs a raw Cypher string against a loaded Kuzu connection and formats the
result rows into the project answer format.
"""

from __future__ import annotations

from typing import Any

import kuzu

from src.core.execution_values import dedupe_rows, format_final_answer, stringify


class CypherExecutor:
    """Execute raw Cypher queries against a Kuzu connection."""

    def format(self, rows: list[list[Any]]) -> str:
        """Dedupe rows and format into the project answer format."""
        deduped = dedupe_rows(rows)
        # An aggregate over an empty set returns NULL; render it consistently
        # with an empty result as "(no result)".
        if deduped and all(cell is None for row in deduped for cell in row):
            return "(no result)"
        return format_final_answer(deduped)

    def execute_raw(self, cypher: str, conn: kuzu.Connection) -> list[list[Any]]:
        """Run a plain Cypher string and return rows."""
        result = conn.execute(cypher)
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(list(result.get_next()))
        return rows


def combine_answers(row_lists: list[list[list[Any]]]) -> str:
    """Cartesian-product the rows of independent sub-questions and render them
    as ``a, b; a, b`` pairs (``;`` separates combined rows, ``,`` separates
    cells).

    Used when Call 1 splits an independent-conjunction question into several
    sub-questions. Each sub-question's rows are combined with every other's,
    and the result follows the project's answer format for paired lists.
    """
    if not row_lists:
        return "(no result)"
    product: list[list[Any]] = [[]]
    for rows in row_lists:
        if not rows:
            return "(no result)"
        product = [combo + list(row) for combo in product for row in rows]
    deduped = dedupe_rows(product)
    if not deduped or all(cell is None for row in deduped for cell in row):
        return "(no result)"
    return "; ".join(", ".join(stringify(cell) for cell in row) for row in deduped)
