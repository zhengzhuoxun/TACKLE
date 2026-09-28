"""
Step 0: Holistic Table Understanding

This step sees all tables together and produces authoritative metadata for:
  - table descriptions
  - column descriptions
  - column semantic types
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import Field

from src.data.loader import QAItem, Table
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import table_understanding as prompts
from src.utils.logging import log
from src.utils.schema import ProjectModel


class ColumnUnderstanding(ProjectModel):
    """Authoritative understanding of one raw table column."""

    column_name: str
    description: str = ""
    semantic_type: str = "categorical"
    sample_values: list[Any] = Field(default_factory=list)
    is_identifier: bool = False
    identifier_kind: str = ""   # "primary_key" | "foreign_key" | ""


class ForeignKey(ProjectModel):
    """One declared foreign-key relationship originating from this table."""

    column: str
    references_table: str
    references_column: str = ""


class TableUnderstanding(ProjectModel):
    """Authoritative understanding of one table and all its columns."""

    table_name: str
    description: str = ""
    columns: list[ColumnUnderstanding] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    def get_column(self, column_name: str) -> ColumnUnderstanding | None:
        """Return the matching column understanding, case-insensitively."""
        key = column_name.lower()
        for column in self.columns:
            if column.column_name.lower() == key:
                return column
        return None


class TableUnderstandingResult(ProjectModel):
    """Holistic understanding over all tables in one QA item."""

    tables: list[TableUnderstanding] = Field(default_factory=list)

    def get_table(self, table_name: str) -> TableUnderstanding | None:
        """Return the matching table understanding, case-insensitively."""
        key = table_name.lower()
        for table in self.tables:
            if table.table_name.lower() == key:
                return table
        return None

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")


class TableUnderstander:
    """Runs one LLM pass over all tables to establish shared schema understanding."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def understand(self, item: QAItem) -> TableUnderstandingResult:
        """Generate authoritative table/column metadata for one QA item."""
        log.info("  [Understand] Analysing %d tables together", item.num_tables)
        user_prompt = prompts.build_user_prompt(item)
        raw = self.llm.chat_json(prompts.SYSTEM_PROMPT, user_prompt)
        result = self._parse(raw, item)
        log.info("  [Understand] → %d table profiles", len(result.tables))
        return result

    @staticmethod
    def _parse(
        raw: dict,
        item: QAItem,
    ) -> TableUnderstandingResult:
        """Validate and normalize the model output against authoritative schema."""
        by_name: dict[str, TableUnderstanding] = {}
        for entry in raw.get("tables", []):
            try:
                understood = TableUnderstanding.model_validate(entry)
            except Exception as exc:
                log.warning("  [Understand] Dropping malformed table understanding: %s", exc)
                continue
            by_name[understood.table_name.lower()] = understood

        tables: list[TableUnderstanding] = []
        for table_name in item.table_names:
            table = item.tables[table_name]
            understood = by_name.get(table_name.lower())
            if understood is None:
                understood = TableUnderstanding(table_name=table_name)

            column_map: dict[str, ColumnUnderstanding] = {
                column.column_name.lower(): column
                for column in understood.columns
            }

            normalized_columns: list[ColumnUnderstanding] = []
            for column_name in table.columns:
                column = column_map.get(column_name.lower())
                if column is None:
                    column = ColumnUnderstanding(column_name=column_name)
                description = column.description.strip()
                normalized_columns.append(
                    ColumnUnderstanding(
                        column_name=column_name,
                        description=description,
                        semantic_type=(column.semantic_type or "categorical").strip() or "categorical",
                        sample_values=TableUnderstander._sample_column_values(
                            table_name=table_name,
                            column_name=column_name,
                            table=table,
                            count=3,
                        ),
                        is_identifier=bool(column.is_identifier),
                        identifier_kind=(column.identifier_kind or "").strip().lower(),
                    )
                )

            table_description = understood.description.strip()

            tables.append(
                TableUnderstanding(
                    table_name=table_name,
                    description=table_description,
                    columns=normalized_columns,
                    foreign_keys=TableUnderstander._normalize_foreign_keys(
                        table_name=table_name,
                        raw_fks=understood.foreign_keys,
                        item=item,
                    ),
                )
            )

        return TableUnderstandingResult(tables=tables)

    @staticmethod
    def _normalize_foreign_keys(
        table_name: str,
        raw_fks: list[ForeignKey],
        item: QAItem,
    ) -> list[ForeignKey]:
        """Validate declared foreign keys, dropping any that reference unknown
        tables or columns."""
        found = TableUnderstander._find_table(item, table_name)
        if found is None:
            return []
        _, table = found
        column_names = {c.lower() for c in table.columns}
        normalized: list[ForeignKey] = []
        for fk in raw_fks:
            if not fk.column:
                continue
            if fk.column.lower() not in column_names:
                log.warning(
                    "  [Understand] Dropping FK '%s.%s': column not in table",
                    table_name,
                    fk.column,
                )
                continue
            if not fk.references_table:
                log.warning(
                    "  [Understand] Dropping FK '%s.%s': missing references_table",
                    table_name,
                    fk.column,
                )
                continue
            ref_found = TableUnderstander._find_table(item, fk.references_table)
            if ref_found is None:
                log.warning(
                    "  [Understand] Dropping FK '%s.%s': unknown references_table '%s'",
                    table_name,
                    fk.column,
                    fk.references_table,
                )
                continue
            ref_name, ref_table = ref_found
            ref_column = fk.references_column
            if not ref_column or ref_column.lower() not in {c.lower() for c in ref_table.columns}:
                log.warning(
                    "  [Understand] Dropping FK '%s.%s': references_column '%s' not in '%s'",
                    table_name,
                    fk.column,
                    ref_column or "<missing>",
                    ref_name,
                )
                continue
            normalized.append(
                ForeignKey(
                    column=fk.column,
                    references_table=ref_name,
                    references_column=ref_column,
                )
            )
        return normalized

    @staticmethod
    def _find_table(item: QAItem, table_name: str) -> tuple[str, Table] | None:
        """Resolve a table case-insensitively, returning (canonical_name, table)."""
        direct = item.tables.get(table_name)
        if direct is not None:
            return table_name, direct
        key = table_name.lower()
        for name in item.table_names:
            if name.lower() == key:
                return name, item.tables[name]
        return None

    @staticmethod
    def _sample_column_values(
        table_name: str,
        column_name: str,
        table: Table,
        count: int,
    ) -> list[Any]:
        """Pick up to three stable pseudo-random distinct sample values for one column."""
        try:
            column_index = table.columns.index(column_name)
        except ValueError:
            return []

        values = [
            row[column_index]
            for row in table.rows
            if column_index < len(row) and not TableUnderstander._is_missing(row[column_index])
        ]
        rng = random.Random(f"{table_name}:{column_name}:{len(values)}")
        order = list(range(len(values)))
        rng.shuffle(order)
        sampled_values: list[Any] = []
        sampled_values.extend(values[idx] for idx in order[:10])
        unique_samples = TableUnderstander._dedupe_values(sampled_values)
        if len(unique_samples) < count:
            sampled_values.extend(values[idx] for idx in order[10:15])
            unique_samples = TableUnderstander._dedupe_values(sampled_values)
        return unique_samples[:count]

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Treat None and blank strings as missing when sampling examples."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    @staticmethod
    def _dedupe_values(values: list[Any]) -> list[Any]:
        """Deduplicate sampled values while preserving order."""
        seen: set[str] = set()
        deduped: list[Any] = []
        for value in values:
            key = repr(value)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped
