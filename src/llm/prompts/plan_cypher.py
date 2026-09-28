"""
Prompt template for Call 2 of the two-step Cypher pipeline: ground + translate.

Takes the KG schema description, one sub-question, and an optional
schema-grounded plan (from Call 1) and asks the LLM to write ONE raw Kuzu
Cypher query. The plan is ADVISORY — if it conflicts with the sub-question or
the schema, the sub-question wins.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:cypher)?\s*|\s*```$", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Remove optional markdown code fences around a returned Cypher query."""
    return _FENCE_RE.sub("", (text or "").strip()).strip()


SYSTEM_PROMPT = """You are an expert Kuzu Cypher query writer.
Return ONLY the Cypher query text — no explanations, no markdown fences, no JSON."""


PROMPT_TEMPLATE = """# TASK
Translate the sub-question into a single Kuzu Cypher query using ONLY the node tables, relationship tables, columns, and relationship directions listed in the schema.

# GROUNDED PLAN (HINT)
A previous analysis step produced this schema-grounded plan. It is a HINT only — if it conflicts with the sub-question or the schema, follow the sub-question.

{plan}

# RULES
- Use ONLY table/column/relationship names from the schema, matching their exact spelling.
- Match a node table as `(v:TableName)`. The node primary key column is `id`.
- Traverse a relationship table exactly in its schema direction: `(a:SourceTable)-[r:RelTable]->(b:TargetTable)`. Never flip a relationship direction.
- Bind each node/relationship variable exactly once. If the answer needs two independent branches from the same node, list them comma-separated in ONE MATCH, each anchored at the same node variable, e.g. `MATCH (n0:track)-[r0:belongs_to_genre]->(n1:genre), (n0:track)-[r1:has_format_type]->(n2:format)`. Never chain the end of one branch into the start of another.
- Backtick-quote any identifier that is not a plain word or is a reserved keyword (for example `order` must be written as `` `order` ``).
- Kuzu dialect: inequality is `<>` (never `!=`); `IN` takes a list; `CONTAINS` for substring; regex is `=~` (only on string columns).
- Date/datetime literals must be written with parentheses: `DATE('2008-12-01')`, never `DATE '2008-12-01'`. Plain string literals (e.g. `'2008-04-30 16:53'`) are also valid for comparing against datetime columns.
- An `ORDER BY` inside a `WITH` must be immediately followed by `SKIP` or `LIMIT` in that same `WITH`. To sort/limit the final result, put `ORDER BY` and `LIMIT` on the final `RETURN` instead.
- No SQL constructs: no window functions (`OVER ... (PARTITION BY ...)`), no SQL cast functions (`TOFLOAT`, `TONUMBER`, `TOINT`). Use `CAST(x AS TYPE)` or plain comparisons.
- Match property values exactly as shown in the samples (case, date format, quotes).
- If a filter value from the plan does not appear verbatim in the schema's sample values, adjust it to the shortest distinctive substring that does (e.g. the plan says "Marshall County" but samples show "50 Marshall" → use `CONTAINS 'Marshall'`).
- Project human-readable answer columns (names/descriptions), not internal ids, unless the question asks for an id/code.
- Return EXACTLY the columns the question asks to see — never extra columns. An aggregate value, a group key, a count, an id, or a code is returned ONLY if the question explicitly asks for it. An aggregate used merely to RANK or SELECT (e.g. "which X has the most Y") is NOT part of the answer and must not be returned.
- Yes/no questions: `RETURN COUNT(*) > 0 AS answer`.
- "How many" questions: `RETURN COUNT(*) AS answer`; use `COUNT(DISTINCT ...)` when the question says distinct/different.
- For "which X ..." questions, return the requested fields (use `DISTINCT` if duplicates should be collapsed).
- Per-group aggregation: `WITH <group columns>, <agg> AS <id> RETURN ...`; a comparison against an aggregate is a WHERE after that WITH.
- "Which X has the highest/lowest number of Y" (argmax over a group): aggregate per X, sort, and take the top, returning ONLY X:
  `MATCH ... WITH x, COUNT(y) AS cnt ORDER BY cnt DESC LIMIT 1 RETURN x`
  Put `ORDER BY` and `LIMIT`/`SKIP` in the SAME clause (both in the WITH, or both on the RETURN). Define any aggregate alias (e.g. `m = MAX(cnt)`) BEFORE you reference it in a WHERE — Kuzu rejects variables that are not yet in scope.
- Return ONE query ending in a RETURN clause. Do not wrap it in code fences.

# GRAPH SCHEMA
{schema_block}

# SUB-QUESTION
{question}

Return only the Cypher query."""


def build_user_prompt(schema_block: str, question: str, plan: str = "") -> str:
    """Fill the Cypher-writing prompt with schema, sub-question, and grounded plan."""
    return (
        PROMPT_TEMPLATE
        .replace("{schema_block}", schema_block)
        .replace("{question}", question)
        .replace("{plan}", (plan or "").strip() or "(none)")
    )


def build_repair_prompt(
    schema_block: str,
    question: str,
    plan: str,
    previous_cypher: str,
    error: str,
) -> str:
    """Append the failed attempt and its error to the base prompt for a retry."""
    base = build_user_prompt(schema_block, question, plan)
    return (
        base
        + "\n\n# YOUR PREVIOUS ATTEMPT\n"
        + previous_cypher
        + "\n\nIt failed with this error:\n"
        + error
        + "\n\nFix the query and return ONLY the corrected Cypher query."
    )
