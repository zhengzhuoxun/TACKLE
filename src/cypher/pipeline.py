"""Two-step Cypher pipeline: decompose (logic) → ground+translate → execute.

Call 1 (semantic grounding) splits the NL question into independent
sub-questions and describes each in free, schema-grounded text: answer entity,
traversal path, filters, projection, aggregation and ordering.
Call 2 (syntactic rendering) turns each sub-question into one raw Kuzu Cypher
query, using the grounding as a hint only. An optional, flag-gated
repair loop re-asks Call 2 with the execution error when a query fails.

Validation is advisory: it is recorded and used to drive repair, but execution
success is the final arbiter.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from src.core.graph_schema_builder import GraphSchema
from src.core.instance_kg_builder import InstanceKG
from src.cypher.cypher_validator import validate_cypher_schema
from src.cypher.direct_schema import build_kg_description, render_kg_description
from src.cypher.executor import CypherExecutor, combine_answers
from src.cypher.loader import load_to_kuzu
from src.data.loader import QAItem
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import plan_cypher, plan_decompose
from src.utils.logging import log
from src.utils.schema import ProjectModel


class CypherRunResult(ProjectModel):
    """Artifacts from running the two-step Cypher path on one QA item."""

    item_id: int
    question: str
    answer: str = ""
    decomposition: dict | None = None  # raw Call-1 output (soft, verbatim)
    sub_questions: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)  # final Cypher per sub-question
    attempts: list[list[dict]] = Field(default_factory=list)  # per sub-question repair history
    issues: list[str] = Field(default_factory=list)  # advisory validation notes
    error: str | None = None


def _parse_decomposition(raw: dict, question: str) -> tuple[list[str], list[str]]:
    """Extract ``(sub_question, plan)`` pairs from the soft decomposition JSON.

    The plan is the schema-grounded description Call 2 renders into Cypher.
    Tolerant: a malformed or empty output degrades to a single sub-question
    equal to the full question (with no plan hint), never fails.
    """
    queries = raw.get("queries") if isinstance(raw, dict) else None
    pairs: list[tuple[str, str]] = []
    if isinstance(queries, list):
        for q in queries:
            if not isinstance(q, dict):
                continue
            sub = str(q.get("sub_question") or q.get("text") or "").strip()
            plan = str(
                q.get("grounding")
                or q.get("plan")
                or q.get("logic")
                or q.get("requirements")
                or ""
            ).strip()
            pairs.append((sub or question, plan))
    if not pairs:
        pairs = [(question, "")]
    return [s for s, _ in pairs], [l for _, l in pairs]


class CypherPipeline:
    """Two-call Cypher path: decompose → ground+translate → execute."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        repair: bool = False,
        repair_attempts: int = 3,
    ):
        self.llm = llm or get_llm()
        self.executor = CypherExecutor()
        self.repair = repair
        self.repair_attempts = max(1, int(repair_attempts))

    def run(
        self,
        schema: GraphSchema,
        kg: InstanceKG,
        item: QAItem,
    ) -> CypherRunResult:
        """Run the full two-call Cypher path for one QA item."""
        try:
            description = build_kg_description(kg, schema)
            schema_prompt = render_kg_description(description)

            raw: dict = {}
            try:
                raw = self.llm.chat_json(
                    plan_decompose.SYSTEM_PROMPT,
                    plan_decompose.build_user_prompt(schema_prompt, item.question),
                )
                if not isinstance(raw, dict):
                    raw = {}
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Decomposition failed for item %d (%s); using the full "
                    "question as a single sub-question",
                    item.id_,
                    exc,
                )
                raw = {}

            sub_questions, plans = _parse_decomposition(raw, item.question)
            conn = load_to_kuzu(kg)

            row_lists: list[list[list[Any]]] = []
            queries: list[str] = []
            attempts: list[list[dict]] = []
            issues: list[str] = []
            error: str | None = None

            for sub, plan in zip(sub_questions, plans):
                rows, cypher, history, issue_list, err = self._run_sub_question(
                    sub, plan, schema_prompt, schema, conn
                )
                queries.append(cypher or "")
                attempts.append(history)
                issues.extend(issue_list)
                if err:
                    error = error or err
                    row_lists.append([])
                else:
                    row_lists.append(rows)

            if error:
                answer = ""
            elif len(sub_questions) == 1:
                answer = self.executor.format(row_lists[0])
            else:
                answer = combine_answers(row_lists)

            return CypherRunResult(
                item_id=item.id_,
                question=item.question,
                answer=answer,
                decomposition=raw or None,
                sub_questions=sub_questions,
                queries=queries,
                attempts=attempts,
                issues=issues,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Cypher pipeline failed for item %d: %s", item.id_, exc)
            return CypherRunResult(
                item_id=item.id_,
                question=item.question,
                error=str(exc),
            )

    def _run_sub_question(
        self,
        sub_question: str,
        plan: str,
        schema_prompt: str,
        schema: GraphSchema,
        conn: Any,
    ) -> tuple[list[list[Any]], str | None, list[dict], list[str], str | None]:
        """Call 2 with an optional repair loop; return
        ``(rows, last_cypher, history, issues, error)``.
        """
        history: list[dict] = []
        issues: list[str] = []
        error_msg: str | None = None
        cypher: str | None = None
        total_attempts = 1 + (self.repair_attempts if self.repair else 0)

        for attempt in range(total_attempts):
            if error_msg:
                prompt = plan_cypher.build_repair_prompt(
                    schema_prompt, sub_question, plan, cypher or "", error_msg
                )
            else:
                prompt = plan_cypher.build_user_prompt(
                    schema_prompt, sub_question, plan
                )
            try:
                raw_text = self.llm.chat(plan_cypher.SYSTEM_PROMPT, prompt)
                cypher = plan_cypher.strip_fences(raw_text)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"LLM call failed: {exc}"
                history.append({"attempt": attempt, "cypher": None, "error": error_msg})
                continue
            if not cypher:
                error_msg = "LLM returned an empty Cypher query"
                history.append({"attempt": attempt, "cypher": None, "error": error_msg})
                continue

            issues = validate_cypher_schema(cypher, schema)
            history.append({"attempt": attempt, "cypher": cypher, "issues": issues})
            log.info("Cypher query (attempt %d):\n%s", attempt, cypher)
            try:
                rows = self.executor.execute_raw(cypher, conn)
                return rows, cypher, history, issues, None
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                log.warning("Cypher execution failed (attempt %d): %s", attempt, error_msg)

        return [], cypher, history, issues, error_msg
