"""
Data loader for MMQA synthesised table-QA datasets.

Each JSON file is an array of items with structure:
{
  "id_": int,
  "Question": str,
  "SQL": str,
  "table_names": [str, ...],
  "tables": [{"table_columns": [...], "table_content": [[...], ...]}, ...],
  "foreign_keys": [str, ...],
  "primary_keys": [str, ...],
  "answer": str
}
"""

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from src.utils.schema import ProjectModel


class Table(ProjectModel):
    """Represents one table: its columns and row data."""
    columns: list[str]
    rows: list[list[Any]] = Field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.columns)

    def column_index(self, col_name: str) -> Optional[int]:
        """Return the index of a column by name (case-insensitive)."""
        col_lower = col_name.lower()
        for i, c in enumerate(self.columns):
            if c.lower() == col_lower:
                return i
        return None

    def get_column_values(self, col_name: str) -> list[Any]:
        """Get all values for a given column."""
        idx = self.column_index(col_name)
        if idx is None:
            return []
        return [row[idx] for row in self.rows]

    def __repr__(self) -> str:
        return f"Table(cols={self.columns}, rows={self.num_rows})"


class QAExample(ProjectModel):
    """One QA example carried by a merged item."""
    source_item_id: int | None = None
    old_item_id: int | None = None
    old_item_index: int | None = None
    question: str = ""
    sql: str = ""
    answer: str = ""
    original_answer: str = ""
    execution_status: str = ""


class QAItem(ProjectModel):
    """One question-answering item with its tables."""
    id_: int
    question: str = ""
    sql: str = ""
    table_names: list[str] = Field(default_factory=list)
    tables: dict[str, Table] = Field(default_factory=dict)           # name -> Table
    foreign_keys: list[str] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)
    answer: str = ""
    qa_pairs: list[QAExample] = Field(default_factory=list)
    is_merged: bool = False
    source_item_ids: list[int] = Field(default_factory=list)
    schema_signature: list[Any] = Field(default_factory=list)
    descriptions: dict | None = None

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def iter_qa_pairs(self) -> list[QAExample]:
        if self.qa_pairs:
            return self.qa_pairs
        return [
            QAExample(
                source_item_id=self.id_,
                question=self.question,
                sql=self.sql,
                answer=self.answer,
                original_answer=self.answer,
                execution_status="ok",
            )
        ]

    def as_question_item(self, qa_pair: QAExample | None = None) -> "QAItem":
        """Create a single-question QAItem from a merged-item QA pair."""
        base = self.model_copy(deep=True)
        if qa_pair is None:
            qa_pair = self.iter_qa_pairs()[0]
        base.question = qa_pair.question or self.question
        base.sql = qa_pair.sql or self.sql
        base.answer = qa_pair.answer
        base.qa_pairs = []
        base.is_merged = False
        return base

    def __repr__(self) -> str:
        return (f"QAItem(id={self.id_}, question={self.question[:60]}..., "
                f"tables={list(self.tables.keys())})")


def _normalize_answer(raw_answer: Any) -> str:
    """Convert dataset answer payloads into the string form used by evaluation."""
    if raw_answer is None:
        return ""
    if isinstance(raw_answer, str):
        return raw_answer
    if isinstance(raw_answer, dict):
        rows = raw_answer.get("data", [])
        if not isinstance(rows, list) or not rows:
            return ""
        if len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1:
            return str(rows[0][0])
        return ", ".join(str(row[0]) for row in rows if isinstance(row, list) and row)
    if isinstance(raw_answer, list):
        return ", ".join(str(v) for v in raw_answer)
    return str(raw_answer)


def load_dataset(json_path: str | Path, max_items: int | None = None) -> list[QAItem]:
    """
    Load the MMQA JSON dataset and return parsed QAItem objects.

    Args:
        json_path: Path to the JSON file.
        max_items: Optional maximum number of items to return.

    Returns:
        List of QAItem objects.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        entries = raw
    else:
        entries = [raw]

    if max_items is not None and max_items >= 0:
        entries = entries[:max_items]

    items: list[QAItem] = []
    for entry in entries:
        # Build table dict
        tables: dict[str, Table] = {}
        for t_name, t_obj in zip(entry.get("table_names", []), entry.get("tables", [])):
            tables[t_name] = Table(
                columns=t_obj["table_columns"],
                rows=t_obj["table_content"],
            )

        qa_pairs_payload = entry.get("qa_pairs", [])
        qa_pairs = [
            QAExample.model_validate({
                "source_item_id": qa.get("source_item_id"),
                "old_item_id": qa.get("old_item_id"),
                "old_item_index": qa.get("old_item_index"),
                "question": qa.get("Question") or qa.get("question") or "",
                "sql": qa.get("SQL") or qa.get("sql") or "",
                "answer": _normalize_answer(qa.get("answer", "")),
                "original_answer": _normalize_answer(qa.get("original_answer", qa.get("answer", ""))),
                "execution_status": qa.get("execution_status", ""),
            })
            for qa in qa_pairs_payload
        ]

        if qa_pairs:
            first_qa = qa_pairs[0]
            question = first_qa.question or entry.get("Question") or entry.get("question") or ""
            sql = first_qa.sql or entry.get("SQL") or entry.get("sql") or ""
            answer = first_qa.answer
        else:
            question = entry.get("Question") or entry.get("question") or ""
            sql = entry.get("SQL") or entry.get("sql") or ""
            answer = _normalize_answer(entry.get("answer", ""))

        items.append(QAItem.model_validate({
            "id_": entry["id_"],
            "question": question,
            "sql": sql,
            "table_names": entry.get("table_names", []),
            "tables": tables,
            "foreign_keys": entry.get("foreign_keys", []),
            "primary_keys": entry.get("primary_keys", []),
            "answer": answer,
            "qa_pairs": qa_pairs,
            "is_merged": bool(qa_pairs) or "qa_pairs" in entry,
            "source_item_ids": entry.get("source_item_ids", []),
            "schema_signature": entry.get("schema_signature", []),
            "descriptions": entry.get("descriptions"),
        }))

    return items


def describe_item(item: QAItem) -> str:
    """Produce a human-readable description of a QAItem for LLM prompts."""
    lines = [f"**Question**: {item.question}"]
    lines.append(f"\n**Tables** ({len(item.tables)}):")
    for tname, table in item.tables.items():
        lines.append(f"  - `{tname}`: columns={table.columns} "
                     f"({table.num_rows} rows)")
    # if item.foreign_keys:
    #     lines.append(f"\n**Foreign Keys**: {item.foreign_keys}")
    # if item.primary_keys:
    #     lines.append(f"**Primary Keys**: {item.primary_keys}")
    return "\n".join(lines)
