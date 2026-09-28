"""Offline stand-in for an LLM provider, for validating the pipeline end to end.

This exists to answer one question: *if the API key and credits were fine,
would the whole sweep run correctly?* It replaces exactly one thing -- the
network call -- and leaves every other code path real: ICL generation, table
verbalization and clarification, the on-disk caches, the thread pools,
embedding, cosine retrieval, answer checking, per-question-type scoring and the
LaTeX aggregation all execute as they do against a live provider.

**Accuracy numbers from a mock run are meaningless.** The generator does not
reason; it produces structurally valid output of the right shape. A mock sweep
proves the plumbing works, never that a method is good. Runs are written to
their own output directory and labelled ``mock-*`` so they cannot be confused
with real results.
"""

from __future__ import annotations

import hashlib
import math
import re

# The verbalization prompt ends with the markdown table; the clarifier and the
# answer prompts have their own recognizable shapes. The mock keys off these so
# each call site gets output of the shape its parser expects.
_MARKDOWN_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def _table_rows(prompt: str) -> tuple[list[str], list[list[str]]]:
    """Recover (columns, rows) from a markdown table embedded in a prompt."""
    columns: list[str] = []
    rows: list[list[str]] = []
    for line in prompt.splitlines():
        match = _MARKDOWN_ROW.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue  # the |---|---| separator
        if not columns:
            columns = cells
        else:
            rows.append(cells)
    return columns, rows


def _verbalize(prompt: str) -> str:
    """Emit one bullet per row naming every column and value.

    This is the shape TabRAG's real verbalizer produces, so the chunker's
    parsing, the row-coverage accounting and the indexed text are all exercised
    against realistic input.
    """
    columns, rows = _table_rows(prompt)
    if not columns or not rows:
        return "- No tabular content was present in the input."
    lines = []
    for index, row in enumerate(rows, start=1):
        parts = [
            f"a {column} of {row[position] if position < len(row) else ''}"
            for position, column in enumerate(columns)
        ]
        lines.append(f"- For row {index}, the data shows {', '.join(parts)}.")
    return "\n".join(lines)


def _summary(prompt: str) -> str:
    columns, rows = _table_rows(prompt)
    return (
        "table_title: Mock table summary\n"
        f"keywords: {', '.join(columns[:5]) if columns else 'n/a'}\n"
        f"content_overview: A table of {len(rows)} records over "
        f"{len(columns)} columns.\n"
        "data_patterns: Values are categorical and numeric identifiers."
    )


def _glossary(prompt: str) -> str:
    columns, _ = _table_rows(prompt)
    if not columns:
        return "No specialized terms were identified."
    return "\n".join(f"{column}: the {column} recorded for the record."
                     for column in columns[:10])


def _answer(prompt: str) -> str:
    """Answer from the retrieved context, so retrieval quality still shows.

    The mock cannot reason, but echoing a value that actually appears in the
    retrieved documents keeps the answer-checking path meaningful: a cell whose
    retrieval surfaced nothing still yields "(no result)" exactly as it would
    with a live model.
    """
    values: list[str] = []
    for line in prompt.splitlines():
        marker = " of "
        if line.lstrip().startswith("- ") and marker in line:
            for chunk in line.split(marker)[1:]:
                value = chunk.split(",")[0].split(".")[0].strip()
                if value:
                    values.append(value)
    if not values:
        return "(no result)"
    # Deterministic pick, so a rerun of the same question gives the same answer.
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return values[seed % len(values)]


def generate(messages: list[dict]) -> str:
    """Produce a synthetic completion shaped like the caller expects."""
    user = ""
    for message in messages:
        if message.get("role") == "user":
            user = str(message.get("content") or "")
    lowered = user.lower()

    if "verbaliz" in lowered or "extract every" in lowered or "every visible cell" in lowered:
        return _verbalize(user)
    if "summary" in lowered and "table" in lowered and "json" not in lowered:
        return _summary(user)
    if "explain" in lowered or "terms" in lowered or "abbreviat" in lowered:
        return _glossary(user)
    if "json" in lowered:
        return '{"cells": []}'
    return _answer(user)


# Embeddings must be deterministic and similarity-bearing, or retrieval would be
# random and the recall numbers meaningless as a plumbing check. Hashed token
# counts give a stable bag-of-words vector: identical text matches itself, and
# texts sharing tokens score higher than texts that do not.
_EMBED_DIM = 256


def embed(text: str) -> list[float]:
    vector = [0.0] * _EMBED_DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % _EMBED_DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
