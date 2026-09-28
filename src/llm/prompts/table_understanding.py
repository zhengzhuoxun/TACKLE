"""
Prompt templates for Step 0: holistic table understanding.

This step sees all tables together and produces authoritative metadata for:
  - each table description
  - each column description
  - each column semantic type
"""

from __future__ import annotations

import json
import random

from src.data.loader import QAItem


SYSTEM_PROMPT = """You are an expert in database schema understanding.

You will be given all tables of one database example together, including:
- table names
- column headers
- a few example rows per table
- (optionally) authoritative dataset-provided table/column descriptions
- (optionally) value-overlap hints about primary/foreign-key candidates

Your job is to build a compact but reliable understanding of:
1. what each table represents
2. what each column means
3. the semantic type of each column
4. which columns are identifiers (primary keys / foreign keys)
5. the foreign-key relationships between tables (the PK-FK links)

Return JSON:
{
  "tables": [
    {
      "table_name": "<table name>",
      "description": "<one-sentence description of what the table stores>",
      "columns": [
        {
          "column_name": "<column name>",
          "description": "<what this column means>",
          "semantic_type": "<numeric | datetime | boolean | string>",
          "is_identifier": <true | false>,
          "identifier_kind": "<primary_key | foreign_key | empty string>"
        }
      ],
      "foreign_keys": [
        {
          "column": "<foreign-key column in THIS table>",
          "references_table": "<the table this column references>",
          "references_column": "<the column in that table it references>"
        }
      ]
    }
  ]
}

Rules:
1. Every input table must appear exactly once.
2. Every input column must appear exactly once under its table.
3. **Foreign keys MUST be declared explicitly in `foreign_keys`, not only in prose.**
   - For every column in a table whose values identify/reference rows of another table (i.e. it joins to that table), add one entry to that table's `foreign_keys` array.
   - `column` is the column in the current table; `references_table` is the referenced table; `references_column` is the referenced column in that table (the two names may differ — the FK column and the referenced PK column can have different names).
   - Direction matters: the FK lives in the child/referencing table and points to the parent/referenced table. Example: `dept_lead_map.dept_ref` references `dept_registry.dept_key`, so `dept_lead_map`'s `foreign_keys` contains `{"column": "dept_ref", "references_table": "dept_registry", "references_column": "dept_key"}` — and `dept_registry` has no entry for `dept_key`.
   - If a column does not reference another table, do not list it.
   - The `foreign_keys` array may be empty for a table.
4. **Identifier marking**:
   - A column that uniquely identifies each row of its own table is a primary key: set `is_identifier=true` and `identifier_kind="primary_key"`.
   - A column that references another table's identifier is a foreign key: set `is_identifier=true` and `identifier_kind="foreign_key"`.
   - All other columns: `is_identifier=false` and `identifier_kind=""`.
5. **`semantic_type` reflects the STORED value type, not the semantic concept.**
   - `boolean` is ONLY for columns whose values are literally `true`/`false` (or `0`/`1`).
   - Columns whose values are words like `yes`/`no`, `y`/`n`, `pass`/`fail`, `accepted`/`rejected` are `string`, NOT `boolean` — the stored values are text even though they mean "yes/no".
6. **Cross-table awareness and signal-word usage**:
   - When a column's values refer to a concept that exists as its own table elsewhere in the database, describe it using the pattern:
     `"Identifies the [concept] (see the [concept] table)."` or
     `"References the [concept] concept introduced in the [table] table."`
   - Even when there is **no separate dedicated table** for that concept, if the column represents a real-world entity that could be shared across multiple tables (e.g., a name, code, or identifier of a category, organization, location, etc.), you **must** still use a signal phrase that downstream clustering steps can reliably detect. **Preferred phrasing:**
     `"Identifies the [concept]."` or `"References the [concept] concept."`
     (Avoid vague terms like "represents" or "indicates" in the description.)
7. Keep descriptions concise and factual.
8. Return ONLY JSON.
"""


def build_user_prompt(
    item: QAItem,
) -> str:
    """Render all tables together for holistic schema understanding."""
    parts = [
        "Understand all tables together before any later clustering or relation reasoning.",
        "",
        "## Tables",
    ]

    for table_name in item.table_names:
        table = item.tables[table_name]
        parts.append(f"### `{table_name}`")
        parts.append(f"Columns: {json.dumps(table.columns)}")
        parts.append("Sample rows (3 randomly picked rows when available):")
        if table.rows:
            for row in _sample_rows(table_name, table.rows, count=3):
                parts.append(f"  {dict(zip(table.columns, row))}")
        else:
            parts.append("  (no rows)")
        parts.append("")

    parts.append(
        "Please provide one description for each table, and for each column provide "
        "a concise description plus semantic type, identifier marks, and the "
        "foreign-key links between tables."
    )
    return "\n".join(parts)


def _sample_rows(table_name: str, rows: list[list[object]], count: int) -> list[list[object]]:
    """Pick a stable pseudo-random row subset for prompt grounding."""
    if len(rows) <= count:
        return rows
    rng = random.Random(f"{table_name}:{len(rows)}")
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[idx] for idx in indices]
