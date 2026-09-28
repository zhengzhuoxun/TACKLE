"""
Prompt template for the direct Cypher translation variant.

One LLM call: the KG schema (nodes/relations with per-property samples) plus
the natural-language question, straight to one raw Kuzu Cypher query. There is
no intent decomposition, grounding, or deterministic composition in this path.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are an expert Kuzu Cypher query writer.
You translate a natural-language question directly into ONE valid Cypher query
against the knowledge-graph schema provided in the user message.
Return ONLY the Cypher query text — no explanations, no markdown fences, no JSON."""


PROMPT_TEMPLATE = """# TASK
Translate the natural-language question into a single Kuzu Cypher query using ONLY the node tables, relationship tables, columns, and relationship directions listed in the schema below.

# RULES
- Use ONLY table/column/relationship names from the schema, matching their exact spelling.
- Match a node table as `(v:TableName)`. The node primary key column is `id`.
- Traverse a relationship table exactly in its schema direction: `(a:SourceTable)-[r:RelTable]->(b:TargetTable)`. Never flip a relationship direction.
- Backtick-quote any identifier that is not a plain word or is a reserved keyword (for example `order` must be written as `` `order` ``).
- Kuzu dialect: inequality is `<>` (never `!=`); `IN` takes a list; `CONTAINS` for substring; regex is `=~` (only on string columns).
- Match property values exactly as shown in the samples (case, date format, quotes).
- Project human-readable answer columns (names/descriptions), not internal ids, unless the question asks for an id/code.
- Yes/no questions: `RETURN COUNT(*) > 0 AS answer`.
- "How many" questions: `RETURN COUNT(*) AS answer`; use `COUNT(DISTINCT ...)` when the question says distinct/different.
- For "which X ..." questions, return the requested fields (and use `DISTINCT` if duplicates should be collapsed).
- For superlative "which X has the most/fewest/highest/lowest/most recent/..." questions (and "top N ..." questions): compute the ranking value first with `WITH x, COUNT(...)/SUM(...)/AVG(...)/MAX(...)/MIN(...) AS rank_value`, then `RETURN` ONLY the entity's identifying field(s) that answer "which X" — do NOT also project `rank_value`. Order by that materialized alias: `ORDER BY rank_value DESC LIMIT 1` (or `LIMIT N`). Kuzu does NOT allow an aggregate function call directly inside `ORDER BY`/`WHERE`/`RETURN` — it must first be bound to a name in a `WITH` clause, or the query fails with "Cannot evaluate expression with type AGGREGATE_FUNCTION".
- Return ONE query ending in a RETURN clause. Do not wrap it in code fences.

# GRAPH SCHEMA
{schema_block}

# QUESTION
{question}

Return only the Cypher query."""


def build_user_prompt(schema_block: str, question: str) -> str:
    """Fill the direct-translation prompt with the schema and question."""
    return (
        PROMPT_TEMPLATE
        .replace("{schema_block}", schema_block)
        .replace("{question}", question)
    )
