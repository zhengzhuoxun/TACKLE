"""
Prompt template for Call 1 of the two-step Cypher pipeline: logical decomposition.

Takes the KG schema description (the same text the direct translator sees) plus
the natural-language question and asks the LLM to split the question into
independent sub-questions, describing the *logic* of each in free text.

The output is a SOFT contract: the next call treats it as a reasoning hint, not
as an authoritative plan. Nothing here is validated or folded into rigid
structures.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a careful schema-grounded query analyser.
Return only valid JSON that follows the requested output shape."""


PROMPT_TEMPLATE = """# ROLE
You are a schema-grounded query analyser. You map the question's meaning onto the concrete schema terms (node tables, relationship tables, columns), so a later step can render Cypher mechanically.

# TASK
Split the question into one or more INDEPENDENT sub-questions. Most questions are a single sub-question; only split when the question asks for several unrelated answers that must be combined (e.g. "give X and give Y").

For each sub-question, write a `grounding` description that pins down, in SCHEMA terms:
- **answer entity**: the schema node table the answer is about.
- **traversal**: the concrete path(s) from the answer entity to every other entity the question mentions, written as `entity -[relationship]-> entity` chains using EXACT schema names and directions. Include every hop needed to reach a filter's entity or the thing being counted.
- **filters**: each condition as `entity.property operator value`, using EXACT column names. Take the VALUE from the question, but align it with the schema's sample values — if the question's phrase does not appear verbatim in any sample (e.g. the question says "Marshall County" but the samples show "50 Marshall"), use the longest distinctive substring that DOES appear (e.g. `CONTAINS 'Marshall'`). Include "exists" / "does not exist" conditions too.
- **projection**: EXACTLY the columns the question asks to SEE, written as `entity.property` — and nothing else. If the question asks "which X", project only X's name; do NOT include an aggregate/count that is used merely to rank or select (e.g. for "highest number of Y", project the X name only, not the count).
- **computation**: any count/sum/avg/min/max, whether global or per-group, and the group key in schema terms.
- **ordering / limit**: any top-N, ranking, or ordering.

# RULES
- Use ONLY node table names, relationship names, and column names from the schema, matching their exact spelling.
- The `grounding` is a HINT for a later Cypher writer, not an executed plan — the writer may correct it. Do NOT write Cypher yourself.
- Do not invent entities, relationships, or values that are not in the question or the schema.
- Align filter VALUES with the schema's sample values: never over-specify a literal the data does not contain (question wording like "Marshall County" may correspond to sample values like "50 Marshall" — then write `CONTAINS 'Marshall'`, not `= 'Marshall County'`).
- If a concept in the question does not map to an obvious schema name (e.g. "gas stations" may actually be represented by a link table, not a station table), resolve it to the schema element that truly represents it and say so.
- If the question has no aggregation, say so plainly.

# OUTPUT FORMAT
Return ONE JSON object:
{
  "reasoning": "short summary of the overall mapping",
  "queries": [
    {"id": "q1", "sub_question": "<the sub-question in your own words>", "grounding": "<free-text schema-grounded description covering answer entity, traversal, filters, projection, computation, ordering>"}
  ]
}

# GRAPH SCHEMA
{schema_block}

# QUESTION
{question}

Return only the JSON object."""


def build_user_prompt(schema_block: str, question: str) -> str:
    """Fill the decomposition prompt with the KG description and question."""
    return (
        PROMPT_TEMPLATE
        .replace("{schema_block}", schema_block)
        .replace("{question}", question)
    )
