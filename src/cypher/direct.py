"""Direct Cypher translation variant (single LLM call, no planning stages).

Pipeline:
  1. Build the Kuzu-visible KG schema description (nodes/relations with up to
     3 sample values per property) from the instance KG.
  2. Ask the LLM to translate the NL question directly into one raw Cypher
     query.
  3. Load the instance KG into Kuzu, run the query, and format the rows with
     the shared :class:`CypherExecutor`.

Reuses :func:`src.cypher.loader.load_to_kuzu` for materialisation and
:class:`src.cypher.executor.CypherExecutor` for execution/formatting.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from src.core.graph_schema_builder import GraphSchema
from src.core.instance_kg_builder import InstanceKG
from src.cypher.direct_schema import build_kg_description, render_kg_description
from src.cypher.executor import CypherExecutor
from src.cypher.loader import load_to_kuzu
from src.data.loader import QAItem
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import direct_cypher
from src.utils.logging import log
from src.utils.schema import ProjectModel


_FENCE_RE = re.compile(r"^```(?:cypher)?\s*|\s*```$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences around a returned Cypher query."""
    return _FENCE_RE.sub("", text.strip()).strip()


class DirectRunResult(ProjectModel):
    """Artifacts from running the direct Cypher translation on one QA item."""

    item_id: int
    question: str
    answer: str = ""
    cypher: str | None = None
    rows: list[list[Any]] = Field(default_factory=list)
    schema_prompt: str = ""
    error: str | None = None


class DirectCypherPipeline:
    """Runs the one-shot NL→Cypher→execute path for one QA item."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()
        self.executor = CypherExecutor()

    def run(
        self,
        schema: GraphSchema,
        kg: InstanceKG,
        item: QAItem,
    ) -> DirectRunResult:
        """Translate and execute one question directly (single pass)."""
        schema_prompt = ""
        cypher: str | None = None
        try:
            description = build_kg_description(kg, schema)
            schema_prompt = render_kg_description(description)
            prompt = direct_cypher.build_user_prompt(schema_prompt, item.question)
            raw = self.llm.chat(direct_cypher.SYSTEM_PROMPT, prompt)
            cypher = _strip_fences(raw)
            if not cypher:
                return DirectRunResult(
                    item_id=item.id_,
                    question=item.question,
                    schema_prompt=schema_prompt,
                    error="LLM returned an empty Cypher query",
                )

            log.info("Direct Cypher query:\n%s", cypher)
            conn = load_to_kuzu(kg)
            rows = self.executor.execute_raw(cypher, conn)
            answer = self.executor.format(rows)
            return DirectRunResult(
                item_id=item.id_,
                question=item.question,
                answer=answer,
                cypher=cypher,
                rows=rows,
                schema_prompt=schema_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Direct Cypher pipeline failed for item %d: %s", item.id_, exc)
            return DirectRunResult(
                item_id=item.id_,
                question=item.question,
                cypher=cypher,
                schema_prompt=schema_prompt,
                error=str(exc),
            )
