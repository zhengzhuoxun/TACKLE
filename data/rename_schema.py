#!/usr/bin/env python3
"""
rename_schema.py — "dirty" (obfuscate) the table/column names of the merged
MMQA dataset using an LLM, without changing data values, questions, or answers.

For each merged item the script:
  1. builds a prompt from the item's table names + columns + up to 3 sample
     values per column,
  2. calls the LLM ONCE to get (a) a short description of every table/column
     and (b) new (noised) names for every table/column,
  3. validates the result deterministically (1:1 coverage, unique column names
     within a table, unique table names, valid identifiers, and EVERY name
     differs from its original),
  4. writes two outputs:
       <base>/dirty_merged_MMQA/dirty_merged_MMQA_<name>.json  (one entry per item)
       <base>/dirty_MMQA/dirty_MMQA_<name>.json                (one entry per QA pair)
     plus a separate mapping JSON in each folder (old -> new renames, keyed by
     item id_ / source_item_id).

SQL / PK / FK / schema_signature are NOT copied to the outputs.

Usage:
  venv/bin/python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json
  venv/bin/python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json --limit 10
  venv/bin/python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json --resume
  venv/bin/python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json --dry-run --out-dir /tmp/test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient  # noqa: E402

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DecimalEncoder(json.JSONEncoder):
    """Encode Decimal (from ijson) back to int/float, matching the source JSON."""

    def default(self, o):
        if isinstance(o, Decimal):
            if o == o.to_integral_value():
                return int(o)
            return float(o)
        return super().default(o)

# Per-item "noise mode" hints.  The empty string means "no hint"; the LLM's
# temperature + the general rules then decide the style.
MODE_HINTS = [
    "",
    "Noise mode for this item: use heavy abbreviations for column names (e.g. student_id -> stu_id).",
    "Noise mode for this item: prefer camelCase / PascalCase renames (e.g. student_id -> studentId).",
    "Noise mode for this item: use a mix of heavy abbreviations and full synonym renames.",
    "Noise mode for this item: replace at least one table name with a generic placeholder such as 'table1'.",
    "Noise mode for this item: prefer short single-word or two-word column names.",
]

SYSTEM_PROMPT = """\
You are an expert at obfuscating relational database schemas for schema-linking benchmark evaluation.

You will be given ONE item containing 2 or 3 tables. For each table you get its name and, for every column, the column name plus up to 3 sample values.

Your job is to return a JSON object that (a) writes a short description of each table and each column, and (b) assigns a NEW name to each table and each column.

Renaming rules:
1. Column names: PRESERVE the meaning, but change the surface form. Use synonyms, abbreviations, or a different formatting convention (studentID -> stu_nr / student_id / Student / std_no). Never change what the column means.
2. Table names: rename with a synonym/reformatted name, or occasionally replace a table name with a GENERIC placeholder such as "table1" or "table2". Every table name MUST differ from its original name.
3. FORGET any primary/foreign-key relationships. Rename every column independently. Do NOT keep matching columns across tables aligned; it is expected and fine that the same concept gets a different name in different tables.
4. Within ONE table all column names must be UNIQUE.
5. All table names within the item must be UNIQUE.
6. EVERY table name and EVERY column name MUST change: the new name must be DIFFERENT from the original name.
7. Vary the renaming STYLE from item to item (heavy abbreviation / light paraphrase / mask tables).
8. Names must be valid identifiers: letters, digits, underscore; must not start with a digit; no spaces.

Return ONLY a JSON object (no markdown fences, no commentary) with this exact shape:
{
  "tables": [
    {
      "new": "<new table name>",
      "description": "<one short sentence>",
      "columns": [
        {"new": "<new column name>", "description": "<one short sentence>"},
        ...
      ]
    },
    ...
  ]
}
Keep tables in the same order as the input, and columns within each table in the same order as the input.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_ident(name) -> bool:
    return isinstance(name, str) and bool(IDENT_RE.match(name))


def _fmt_value(value) -> str:
    if isinstance(value, str):
        s = value if len(value) <= 30 else value[:30] + "..."
        return f'"{s}"'
    return str(value)


def _infer_type(values) -> str:
    has_real = False
    for v in values:
        if isinstance(v, bool):
            return "BOOLEAN"
        if isinstance(v, Decimal):
            if v != v.to_integral_value():
                has_real = True
            continue
        if isinstance(v, int):
            continue
        if isinstance(v, float):
            has_real = True
            continue
        if isinstance(v, str):
            return "TEXT"
    return "REAL" if has_real else "INTEGER"


def _column_samples(table) -> list[list]:
    """Up to 3 distinct, non-empty sample values per column, in column order."""
    cols = table.get("table_columns", [])
    rows = table.get("table_content", [])
    out: list[list] = []
    for j in range(len(cols)):
        seen = []
        for row in rows:
            if not isinstance(row, list) or j >= len(row):
                continue
            v = row[j]
            if v is None or v == "":
                continue
            if v not in seen:
                seen.append(v)
            if len(seen) >= 3:
                break
        out.append(seen)
    return out


def _render_prompt(table_names, tables, mode_hint: str) -> str:
    parts = []
    for i, table in enumerate(tables):
        name = table_names[i] if i < len(table_names) else f"table_{i + 1}"
        cols = table.get("table_columns", [])
        samples = _column_samples(table)
        lines = [
            f"TABLE {i + 1}: {name}",
            "Columns (name | type | up to 3 sample values):",
        ]
        for j, col in enumerate(cols):
            vals = samples[j]
            typ = _infer_type(vals)
            sample_str = ", ".join(_fmt_value(v) for v in vals) if vals else "(no values)"
            lines.append(f"  {col} | {typ} | {sample_str}")
        parts.append("\n".join(lines))
    body = "\n\n".join(parts)
    if mode_hint:
        body += "\n\n" + mode_hint
    return body


def _validate(resp, table_names, tables) -> str | None:
    """Return an error string if the LLM response is unusable, else None."""
    if not isinstance(resp, dict):
        return "response is not a JSON object"
    entries = resp.get("tables")
    if not isinstance(entries, list) or len(entries) != len(table_names):
        got = len(entries) if isinstance(entries, list) else type(entries).__name__
        return f"expected {len(table_names)} table entries but got {got}"
    seen_tables = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return f"table entry {i} is not an object"
        new_name = entry.get("new")
        if not _valid_ident(new_name):
            return f"invalid new table name {new_name!r} for original {table_names[i]!r}"
        if new_name == table_names[i]:
            return f"table name {new_name!r} was not changed from original {table_names[i]!r}"
        if new_name in seen_tables:
            return f"duplicate table name {new_name!r}"
        seen_tables.add(new_name)
        cols = entry.get("columns")
        expected = tables[i].get("table_columns", [])
        if not isinstance(cols, list) or len(cols) != len(expected):
            got = len(cols) if isinstance(cols, list) else type(cols).__name__
            return f"table {table_names[i]!r}: expected {len(expected)} column entries but got {got}"
        seen_cols = set()
        for j, c in enumerate(cols):
            if not isinstance(c, dict):
                return f"table {table_names[i]!r}: column entry {j} is not an object"
            cn = c.get("new")
            if not _valid_ident(cn):
                return f"invalid new column name {cn!r} for original {expected[j]!r}"
            if cn == expected[j]:
                return f"column name {cn!r} was not changed from original {expected[j]!r} in table {table_names[i]!r}"
            if cn in seen_cols:
                return f"duplicate column name {cn!r} in table {table_names[i]!r}"
            seen_cols.add(cn)
    return None


def _fallback(table_names, tables) -> dict:
    """Deterministic last-resort renaming used when the LLM keeps failing."""
    new_names = [f"table{i + 1}" for i in range(len(table_names))]
    entries = []
    for i, table in enumerate(tables):
        cols = table.get("table_columns", [])
        entries.append(
            {
                "new": new_names[i],
                "description": "auto-renamed (LLM fallback)",
                "columns": [
                    {"new": f"col{j + 1}", "description": "auto-renamed (LLM fallback)"}
                    for j in range(len(cols))
                ],
            }
        )
    return {"tables": entries}


def _rename_item(client, table_names, tables, mode_hint: str, max_retries: int):
    """Call the LLM (with retries).  Returns (response, error_or_None)."""
    user = _render_prompt(table_names, tables, mode_hint)
    feedback = ""
    for _attempt in range(max_retries + 1):
        prompt = user
        if feedback:
            prompt += f"\n\nYour previous response was rejected. Fix this problem and return the corrected JSON: {feedback}"
        try:
            resp = client.chat_json(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 - report any LLM/parse error
            feedback = f"could not parse the JSON output ({exc})"
            time.sleep(1.0)
            continue
        err = _validate(resp, table_names, tables)
        if err is None:
            return resp, None
        feedback = err
        time.sleep(1.0)
    return None, feedback


def _apply(resp: dict, item: dict):
    """Turn a validated response into noised tables + descriptions + mapping."""
    table_names = item.get("table_names", [])
    tables = item.get("tables", [])

    new_names = [e["new"] for e in resp["tables"]]
    new_tables = []
    for i, table in enumerate(tables):
        new_cols = [c["new"] for c in resp["tables"][i]["columns"]]
        new_tables.append(
            {"table_columns": new_cols, "table_content": table.get("table_content", [])}
        )

    descriptions = {"tables": [], "columns": []}
    for i, e in enumerate(resp["tables"]):
        descriptions["tables"].append(
            {"name": e["new"], "description": e.get("description", "")}
        )
        for c in e["columns"]:
            descriptions["columns"].append(
                {"table": e["new"], "name": c["new"], "description": c.get("description", "")}
            )

    renames = []
    for i, (old_t, new_t) in enumerate(zip(table_names, new_names)):
        renames.append({"type": "table", "old": old_t, "new": new_t})
        for old_c, c in zip(
            tables[i].get("table_columns", []),
            resp["tables"][i]["columns"],
        ):
            renames.append(
                {"type": "column", "table": old_t, "old": old_c, "new": c["new"]}
            )
    mapping = {"renames": renames}
    return new_names, new_tables, descriptions, mapping


def _clean_answer(answer):
    """Golden answers: keep only the data (list answers store old column names
    in a 'columns' field, which we drop)."""
    if isinstance(answer, dict):
        return answer.get("data", [])
    return answer


# ---------------------------------------------------------------------------
# Streaming JSON-array writer with resume support
# ---------------------------------------------------------------------------
class ArrayStream:
    """Writes JSON objects into a top-level array, streaming."""

    def __init__(self, path: Path, resume: bool = False):
        self.path = Path(path)
        self._first = True
        if resume and self.path.exists():
            with open(self.path, "rb+") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    f.write(b"[")
                else:
                    f.seek(max(0, size - 64))
                    tail = f.read()
                    pos = tail.rfind(b"]")
                    if pos == -1:
                        raise ValueError(f"Cannot resume {path}: no closing ']' found")
                    truncate_at = max(0, size - len(tail)) + pos
                    f.seek(truncate_at)
                    f.truncate()
                    self._first = truncate_at <= 1
            self.f = open(self.path, "a", encoding="utf-8")
        else:
            self.f = open(self.path, "w", encoding="utf-8")
            self.f.write("[")
            self._first = True

    def write(self, obj: dict) -> None:
        if not self._first:
            self.f.write(",")
        self.f.write("\n")
        json.dump(obj, self.f, ensure_ascii=False, cls=DecimalEncoder)
        self._first = False

    def close(self) -> None:
        self.f.write("\n]\n")
        self.f.close()


def _write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Obfuscate merged-MMQA table/column names via an LLM."
    )
    p.add_argument(
        "--input",
        required=True,
        help="path to merged_two_table.json or merged_three_table.json",
    )
    p.add_argument("--start", type=int, default=0, help="0-based index of first item")
    p.add_argument("--end", type=int, default=3, help="0-based index of last item (inclusive)")
    p.add_argument("--limit", type=int, default=None, help="max number of items to process")
    p.add_argument("--seed", type=int, default=0, help="seed for per-item mode selection")
    p.add_argument("--temperature", type=float, default=0.9, help="LLM temperature override")
    p.add_argument("--max-retries", type=int, default=2, help="LLM retries after validation failure")
    p.add_argument("--resume", action="store_true", help="skip items already in the mapping file")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use deterministic fallback renaming (no LLM) to test I/O",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="base dir containing dirty_merged_MMQA/ and dirty_MMQA/ (default: input folder)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # input_path = Path(args.input)


    if args.input == "two_table":
        input_path = Path("data/merged_MMQA/merged_two_table.json")
    elif args.input == "three_table":
        input_path = Path("data/merged_MMQA/merged_three_table.json")

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    name = "three_table" if "three" in input_path.stem else "two_table"
    base = Path(args.out_dir) if args.out_dir else input_path.parent
    merged_dir = base / "dirty_merged_MMQA"
    split_dir = base / "dirty_MMQA"
    merged_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    merged_out = merged_dir / f"dirty_merged_MMQA_{name}.json"
    split_out = split_dir / f"dirty_MMQA_{name}.json"
    merged_map_path = merged_dir / f"mapping_{name}.json"
    split_map_path = split_dir / f"mapping_{name}.json"

    merged_mapping: dict = {}
    split_mapping: dict = {}
    if args.resume:
        if merged_map_path.exists():
            merged_mapping = json.loads(merged_map_path.read_text(encoding="utf-8"))
        if split_map_path.exists():
            split_mapping = json.loads(split_map_path.read_text(encoding="utf-8"))
        print(f"resume: {len(merged_mapping)} items already in mapping")

    client = None
    if not args.dry_run:
        client = LLMClient()
        client.temperature = args.temperature

    mw = ArrayStream(merged_out, resume=args.resume)
    sw = ArrayStream(split_out, resume=args.resume)

    written = skipped = failed = 0
    idx = 0

    try:
        with open(input_path, "rb") as fh:
            for item in ijson.items(fh, "item"):
                if idx < args.start:
                    idx += 1
                    continue
                if args.end is not None and idx > args.end:
                    break
                if args.limit is not None and written >= args.limit:
                    break

                item_id = item.get("id_")
                key = str(item_id)
                if args.resume and key in merged_mapping:
                    skipped += 1
                    idx += 1
                    continue

                table_names = item.get("table_names", [])
                tables = item.get("tables", [])

                mode_hint = MODE_HINTS[
                    random.Random(f"{args.seed}:{item_id}").randrange(len(MODE_HINTS))
                ]

                fallback_used = False
                if args.dry_run:
                    resp = _fallback(table_names, tables)
                else:
                    resp, err = _rename_item(
                        client, table_names, tables, mode_hint, args.max_retries
                    )
                    if resp is None:
                        resp = _fallback(table_names, tables)
                        fallback_used = True

                try:
                    new_names, new_tables, descriptions, mapping = _apply(resp, item)
                except Exception as exc:  # noqa: BLE001
                    print(f"item {item_id}: apply error {exc}; skipping", file=sys.stderr)
                    failed += 1
                    idx += 1
                    continue

                qa_pairs_out = []
                for qa in item.get("qa_pairs", []):
                    qa_pairs_out.append(
                        {
                            "source_item_id": qa.get("source_item_id"),
                            "Question": qa.get("Question"),
                            "answer": _clean_answer(qa.get("answer")),
                        }
                    )

                merged_obj = {
                    "id_": item_id,
                    "table_names": new_names,
                    "tables": new_tables,
                    "descriptions": descriptions,
                    "qa_pairs": qa_pairs_out,
                }
                mw.write(merged_obj)

                for qa in qa_pairs_out:
                    split_obj = {
                        "id_": qa["source_item_id"],
                        "table_names": new_names,
                        "tables": new_tables,
                        "descriptions": descriptions,
                        "Question": qa["Question"],
                        "answer": qa["answer"],
                    }
                    sw.write(split_obj)
                    split_mapping[str(qa["source_item_id"])] = mapping

                merged_mapping[key] = mapping
                written += 1
                _write_json(merged_map_path, merged_mapping)
                _write_json(split_map_path, split_mapping)
                if fallback_used:
                    print(f"item {item_id}: LLM failed validation -> deterministic fallback", file=sys.stderr)
                print(f"[{written}] item {item_id} processed ({len(qa_pairs_out)} QA pairs)", flush=True)
                idx += 1
    finally:
        mw.close()
        sw.close()

    print(
        f"done: {written} written, {skipped} skipped (resume), {failed} failed. "
        f"outputs: {merged_out}, {split_out}\n"
        f"mappings: {merged_map_path}, {split_map_path}"
    )


if __name__ == "__main__":
    main()
