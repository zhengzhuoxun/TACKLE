"""
Pipeline Orchestrator

Coordinates the TACKLE pipeline:
  Stage 1 — Semantic Graph Builder  (tables → conceptual KG schema)
  Stage 3 — Instance KG Generator    (KG schema + data → instance KG)
  Stage 4 — Cypher Translation       (question + KG → Cypher → answer), either the
            two-step decompose/render pipeline or the direct 1-call variant

Also handles evaluation: comparing predicted answers against ground truth.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from config.settings import _resolve_phase

from src.data.loader import QAExample, QAItem, load_dataset
from src.core.graph_schema_builder import GraphSchema
from src.core.instance_kg_builder import InstanceKG
from src.cypher.direct import DirectCypherPipeline, DirectRunResult
from src.cypher.pipeline import CypherPipeline, CypherRunResult
from src.graph.builder import SemanticGraphBuilder
from src.graph.instance_builder import InstanceGraphBuilder
from src.llm.client import LLMClient, get_llm
from src.utils.logging import log
from src.utils.schema import ProjectModel

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class StageResult(ProjectModel):
    """Result of running one QA item through the pipeline."""
    item_id: int
    question: str
    ground_truth: str
    predicted: str
    correct: bool
    predicted_direct: str | None = None  # direct 1-call answer (set only when both paths ran)
    correct_direct: bool | None = None   # direct 1-call correctness (set only when both paths ran)
    graph_schema: GraphSchema | None = None
    instance_kg: InstanceKG | None = None
    elapsed_sec: float = 0.0
    error: str | None = None
    source_item_id: int | None = None
    source_item_index: int | None = None
    question_index: int | None = None


class PipelineReport(ProjectModel):
    """Aggregate evaluation report."""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    total_time_sec: float = 0.0
    results: list[StageResult] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "total_time_sec": round(self.total_time_sec, 2),
            "avg_time_sec": round(self.total_time_sec / max(self.total, 1), 2),
        }


# ---------------------------------------------------------------------------
# Answer normalization helpers
# ---------------------------------------------------------------------------

def _safe_literal(value: str):
    """Parse a string as a Python literal, returning None on failure."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


_ANSWER_SEPARATORS = re.compile(r"[,;|]")


def _clean_token(token: str) -> str:
    """Normalise a single answer token for comparison."""
    return (token or "").strip().lower().rstrip(".")


def _flatten_tokens(value, split_scalars: bool = False) -> set[str]:
    """Flatten a (possibly nested) literal into normalised string tokens.

    With ``split_scalars``, each scalar is additionally split on the answer
    separators (`,`, `;`, `|`) so a structured ground truth can be compared
    against the executor's flat ``", "``-joined predicted answer (which
    fragments values that themselves contain commas, e.g. ``"2,000 Man"``).
    """
    tokens: set[str] = set()

    def walk(item) -> None:
        if isinstance(item, (list, tuple, set)):
            for sub in item:
                walk(sub)
        elif item is None:
            return
        else:
            text = str(item)
            if split_scalars:
                for frag in _ANSWER_SEPARATORS.split(text):
                    cleaned = _clean_token(frag)
                    if cleaned:
                        tokens.add(cleaned)
            else:
                cleaned = _clean_token(text)
                if cleaned:
                    tokens.add(cleaned)

    walk(value)
    return tokens


def _answer_set(answer: str) -> set[str]:
    """Reduce an answer string to a set of normalised tokens.

    Handles Python list/tuple literals (e.g. ``"['a', 'b']"`` or
    ``"['Jeff Maggert'], ['Billy Mayfair']"``) as well as loose
    ``|``/``;``/``,`` separated values. ``|`` and ``;`` are hard separators;
    ``,`` is only used as a separator when neither is present, so commas
    inside elements (e.g. ``"Last, First"``) are preserved.
    """
    raw = (answer or "").strip()
    if not raw:
        return set()

    # Whole-string Python container literal (most ground truths). Only bracket-
    # leading strings are treated as literals: ast.literal_eval is lenient about
    # trailing "#" comments, so a flat answer like '"40", #1 Zero, ...' would
    # otherwise be truncated to just '"40"'.
    if raw[0] in ("[", "(", "{"):
        value = _safe_literal(raw)
        if value is not None:
            return _flatten_tokens(value)

    # Loose format: split into elements, parsing each element as a
    # Python literal when possible.
    loose = raw.replace("|", ";")
    parts = loose.split(";") if ";" in loose else loose.split(",")
    tokens: set[str] = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        parsed = _safe_literal(part)
        if parsed is not None:
            tokens |= _flatten_tokens(parsed)
        else:
            cleaned = _clean_token(part)
            if cleaned:
                tokens.add(cleaned)
    return tokens


def _is_literal(answer: str) -> bool:
    """True when the answer is a whole-string Python container literal.

    Only bracket-leading strings qualify: ``ast.literal_eval`` is lenient about
    trailing ``#`` comments, so a flat answer like ``"40", #1 Zero, ...`` would
    otherwise parse as a truncated tuple instead of a loose string.
    """
    raw = (answer or "").strip()
    return bool(raw) and raw[0] in ("[", "(", "{") and _safe_literal(raw) is not None


def _answer_set_flat(answer: str) -> set[str]:
    """Reduce an answer to separator-split tokens at the executor's granularity.

    Unlike :func:`_answer_set`, list-literal scalars are also split on
    `,`/`;`/`|` so a structured ground truth (``['2,000 Man'], ['Rock']``) can
    be compared with the executor's flat ``", "``-joined predicted answer
    (``"2,000 Man, Rock"``), where the comma inside a value is indistinguishable
    from a separator.
    """
    raw = (answer or "").strip()
    if not raw:
        return set()
    if raw[0] in ("[", "(", "{"):
        value = _safe_literal(raw)
        if value is not None:
            return _flatten_tokens(value, split_scalars=True)
    tokens: set[str] = set()
    for frag in _ANSWER_SEPARATORS.split(raw):
        cleaned = _clean_token(frag)
        if cleaned:
            tokens.add(cleaned)
    return tokens


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Runs the full four-stage pipeline end-to-end."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        output_dir: str | Path = "outputs",
        use_direct: bool = False,
        cypher_repair: bool = False,
        cypher_repair_attempts: int = 3,
    ):
        self.llm = llm or get_llm()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage handlers
        self.builder = SemanticGraphBuilder(self.llm)
        self.instance_builder = InstanceGraphBuilder(self.llm)
        self.use_direct = use_direct
        self.cypher_pipeline = CypherPipeline(
            self.llm,
            repair=cypher_repair,
            repair_attempts=cypher_repair_attempts,
        )
        self.direct_pipeline = DirectCypherPipeline(self.llm) if use_direct else None

    # ------------------------------------------------------------------
    # Run a single item
    # ------------------------------------------------------------------
    def run_one(
        self,
        item: QAItem,
        verbose: bool = False,
        phase: str | int = "schema",
    ) -> StageResult:
        """Run the pipeline on one QA item; return the result."""
        if item.is_merged:
            return self.run_merged_item(item, verbose=verbose, phase=phase)[0]
        return self._run_single_question(
            item=item,
            verbose=verbose,
            phase=phase,
            item_dir=self.output_dir / f"item_{item.id_:04d}",
        )

    def run_merged_item(
        self,
        item: QAItem,
        verbose: bool = False,
        phase: str | int = "schema",
    ) -> list[StageResult]:
        """Run shared schema-building once per merged item, then process each QA pair."""
        qa_pairs = item.iter_qa_pairs()
        if not qa_pairs:
            qa_pairs = [QAExample(question=item.question, sql=item.sql, answer=item.answer, original_answer=item.answer)]

        shared_item_dir = self.output_dir / f"item_{item.id_:04d}"
        shared_item_dir.mkdir(parents=True, exist_ok=True)

        stop_at = _resolve_phase(phase)
        shared_schema: GraphSchema | None = None
        shared_instance_kg: InstanceKG | None = None
        kg_error: str | None = None

        if stop_at >= 4:
            try:
                log.info("Stage 1 | Building shared semantic graph for merged item %d ...", item.id_)
                shared_schema = self.builder.build(item, phase=phase)  # type: ignore[assignment]
            except Exception as exc:
                kg_error = f"KG generation failed for merged item {item.id_}: {exc}"
                log.error(kg_error)

        if stop_at >= 5 and kg_error is None:
            if shared_schema is None:
                kg_error = f"KG generation failed for merged item {item.id_}: no schema produced"
                log.error(kg_error)
            else:
                try:
                    if not self.builder.clusterings or not self.builder.merged_classes or not self.builder.relations:
                        raise ValueError("Cannot build instance KG without clustering, merge, and relation outputs")
                    log.info("Stage 3 | Building shared instance-level KG for merged item %d ...", item.id_)
                    shared_instance_kg = self.instance_builder.build(
                        schema=shared_schema,
                        item=item,
                        clusterings=self.builder.clusterings,
                        merged=self.builder.merged_classes,
                        relations=self.builder.relations,
                        understanding=self.builder.table_understanding,
                    )
                except Exception as exc:
                    kg_error = f"Instance KG generation failed for merged item {item.id_}: {exc}"
                    log.error(kg_error)

        # Relaxed abort: when the shared KG cannot be built, do not crash the
        # whole run. Report each QA pair of this item as failed so downstream
        # items still run and the final report accounts for them.
        if kg_error is not None:
            log.warning(
                "Skipping %d QA pair(s) for merged item %d due to KG failure",
                len(qa_pairs),
                item.id_,
            )
            failed_results: list[StageResult] = []
            for idx, qa_pair in enumerate(qa_pairs):
                output_item_id = self._resolve_output_item_id(item, qa_pair)
                failed_results.append(
                    StageResult(
                        item_id=output_item_id if output_item_id is not None else item.id_,
                        question=qa_pair.question or item.question,
                        ground_truth=qa_pair.answer or item.answer,
                        predicted=f"[ERROR] {kg_error}",
                        correct=False,
                        elapsed_sec=0.0,
                        error=kg_error,
                        source_item_id=qa_pair.source_item_id,
                        source_item_index=qa_pair.old_item_index,
                        question_index=idx,
                    )
                )
            return failed_results

        results: list[StageResult] = []
        for idx, qa_pair in enumerate(qa_pairs):
            question_item = item.as_question_item(qa_pair)
            output_item_id = self._resolve_output_item_id(item, qa_pair)
            question_dir = shared_item_dir / f"qa_{self._format_qa_folder_suffix(output_item_id)}"
            result = self._run_single_question(
                item=question_item,
                verbose=verbose,
                phase=phase,
                item_dir=question_dir,
                prebuilt_schema=shared_schema,
                prebuilt_instance_kg=shared_instance_kg,
                source_item_id=qa_pair.source_item_id,
                source_item_index=qa_pair.old_item_index,
                question_index=idx,
                output_item_id=output_item_id,
            )
            results.append(result)

        return results

    def _run_single_question(
        self,
        item: QAItem,
        verbose: bool = False,
        phase: str | int = "schema",
        item_dir: Path | None = None,
        prebuilt_schema: GraphSchema | None = None,
        prebuilt_instance_kg: InstanceKG | None = None,
        source_item_id: int | None = None,
        source_item_index: int | None = None,
        question_index: int | None = None,
        output_item_id: int | None = None,
    ) -> StageResult:
        """Run the pipeline on one QA item; return the result."""
        stop_at = _resolve_phase(phase)
        is_partial = stop_at < 4

        t0 = time.perf_counter()
        schema: GraphSchema | None = prebuilt_schema
        instance_kg: InstanceKG | None = prebuilt_instance_kg
        cypher_result: CypherRunResult | None = None
        direct_result: DirectRunResult | None = None
        predicted = ""
        error = None

        item_dir = item_dir or self.output_dir / f"item_{item.id_:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            if prebuilt_schema is None:
                log.info("Stage 1 | Building semantic graph (phase=%s) ...", phase)
                result = self.builder.build(item, phase=phase)

                if stop_at < 4:
                    schema = None
                    if verbose:
                        self._report_partial(item, phase, item_dir)
                else:
                    schema = result  # type: ignore[assignment]
                    if verbose:
                        self._report_schema(item, schema, item_dir)
            elif verbose and stop_at < 4:
                self._report_partial(item, phase, item_dir)
            elif verbose and stop_at >= 4:
                self._report_schema(item, schema, item_dir)

            if stop_at >= 5:
                if schema is None:
                    raise ValueError("Cannot build instance KG without schema")
                if prebuilt_instance_kg is None:
                    if not self.builder.clusterings or not self.builder.merged_classes or not self.builder.relations:
                        raise ValueError("Cannot build instance KG without clustering, merge, and relation outputs")
                    log.info("Stage 3 | Building instance-level KG ...")
                    instance_kg = self.instance_builder.build(
                        schema=schema,
                        item=item,
                        clusterings=self.builder.clusterings,
                        merged=self.builder.merged_classes,
                        relations=self.builder.relations,
                        understanding=self.builder.table_understanding,
                    )
                if instance_kg is not None:
                    self._save_instance_kg(instance_kg, item_dir)
                    if verbose:
                        self._report_stage3_instance(item, instance_kg, item_dir)

            if stop_at >= 6:
                # The two-step pipeline always runs; the direct 1-call variant
                # (TACKLE-naive) optionally runs alongside it for comparison.
                if schema is None or instance_kg is None:
                    raise ValueError(
                        "Cannot run Cypher path without schema and instance KG"
                    )
                # Plan only through relations the instance KG actually
                # materialised — the raw schema can contain dead edges with no
                # backing rows.
                effective_schema = instance_kg.materialized_schema()

                if self.use_direct:
                    log.info("Stage 4 | Running direct Cypher translation ...")
                    direct_result = self.direct_pipeline.run(effective_schema, instance_kg, item)
                    self._save_direct_pipeline(direct_result, item_dir)
                    if verbose:
                        self._report_direct(item, direct_result, item_dir)

                log.info("Stage 4 | Running Cypher pipeline ...")
                cypher_result = self.cypher_pipeline.run(effective_schema, instance_kg, item)
                self._save_cypher_pipeline(cypher_result, item_dir)
                if verbose:
                    self._report_cypher(item, cypher_result, item_dir)

                predicted = (
                    cypher_result.answer
                    if cypher_result.error is None
                    else f"[ERROR] {cypher_result.error}"
                )

        except Exception as exc:
            log.error("Pipeline failed for item %d: %s", item.id_, exc)
            error = str(exc)
            predicted = f"[ERROR] {exc}"

        elapsed = time.perf_counter() - t0
        correct = self._check_answer(predicted, item.answer)

        predicted_direct: str | None = None
        correct_direct: bool | None = None
        extra = None
        if self.use_direct and direct_result is not None:
            predicted_direct = (
                direct_result.answer
                if direct_result.error is None
                else f"[ERROR] {direct_result.error}"
            )
            correct_direct = self._check_answer(predicted_direct, item.answer)
            extra = {
                "predicted_direct": predicted_direct,
                "correct_direct": correct_direct,
                "predicted_cypher": predicted,
                "correct_cypher": correct,
            }

        self._save_result_summary(
            item=item,
            predicted=predicted,
            correct=correct,
            elapsed=elapsed,
            error=error,
            item_dir=item_dir,
            source_item_id=source_item_id,
            source_item_index=source_item_index,
            question_index=question_index,
            output_item_id=output_item_id,
            extra=extra,
        )
        self._save_intermediate_json(
            item=item,
            schema=schema,
            instance_kg=instance_kg,
            predicted=predicted,
            correct=correct,
            elapsed=elapsed,
            error=error,
            item_dir=item_dir,
            source_item_id=source_item_id,
            source_item_index=source_item_index,
            question_index=question_index,
            output_item_id=output_item_id,
        )

        if verbose:
            log.info(
                "Verbose run complete: predicted=%s correct=%s",
                predicted,
                correct,
            )

        return StageResult(
            item_id=output_item_id if output_item_id is not None else item.id_,
            question=item.question,
            ground_truth=item.answer,
            predicted=predicted,
            correct=correct,
            predicted_direct=predicted_direct,
            correct_direct=correct_direct,
            graph_schema=schema,
            instance_kg=instance_kg,
            elapsed_sec=elapsed,
            error=error,
            source_item_id=source_item_id,
            source_item_index=source_item_index,
            question_index=question_index,
        )

    # ------------------------------------------------------------------
    # Verbose reporting helpers
    # ------------------------------------------------------------------
    def _report_schema(
        self, item: QAItem, schema: GraphSchema, item_dir: Path
    ) -> None:
        """Log graph-schema output to console only (artifacts live in intermediate.json)."""
        header = f"=== STAGE 1: Graph Schema (item {item.id_}) ==="
        body = schema.to_prompt_text()
        log.info("%s\n%s\n%s", header, body, "=" * 50)

    def _report_partial(
        self,
        item: QAItem,
        phase: str | int,
        item_dir: Path,
    ) -> None:
        """Log partial (early-stop) results to console and write to file."""
        phase_str = str(phase)
        header = f"=== PARTIAL RUN (phase={phase_str}) — Item {item.id_} ==="
        lines = [header, ""]

        if self.builder.table_understanding:
            lines.append("--- Table Understanding ---")
            for table in self.builder.table_understanding.tables:
                lines.append(f"  Table: `{table.table_name}`  —  {table.description}")
                for column in table.columns:
                    lines.append(
                        f"    {column.column_name}: type={column.semantic_type}  desc={column.description}"
                    )
                lines.append("")

        # Clusters (always available in partial mode)
        if self.builder.clusterings:
            lines.append("--- Column Clusters ---")
            for tname, cres in self.builder.clusterings.items():
                lines.append(f"  Table: `{tname}`  —  {cres.table_description}")
                for cl in cres.clusters:
                    lines.append(
                        f"    [{cl.cluster_id}]  {cl.label:<25s}  "
                        f"pk={cl.primary_key_candidate:<20s}  "
                        f"cols={cl.columns}"
                    )
                for prop in cres.relation_properties:
                    lines.append(
                        f"    [relprop] {prop.column_name:<18s} "
                        f"depends_on={prop.depends_on}  desc={prop.description}"
                    )
                lines.append("")

        # Merged classes (if phase >= merge)
        if self.builder.merged_classes:
            lines.append("--- Merged Classes ---")
            for cls in self.builder.merged_classes.classes:
                lines.append(f"  [{cls.class_id}]  {cls.label}  pk={cls.primary_key}")
            lines.append("")

        # Relations (if phase >= relation)
        if self.builder.relations:
            lines.append("--- Relations ---")
            for rel in self.builder.relations.relations:
                lines.append(
                    f"  {rel.source_class} --[{rel.relation}]--> {rel.target_class} "
                    f"props={[prop.name for prop in rel.properties]}"
                )
            lines.append("")

        body = "\n".join(lines)
        log.info("%s", body)

    def _write_verbose_report_partial(
        self,
        item: QAItem,
        phase: str | int,
        elapsed: float,
        error: str | None,
        item_dir: Path,
    ) -> None:
        """Write a partial (early-stop) verbose report."""
        lines = []
        sep = "=" * 70
        lines.append(sep)
        lines.append(f"PARTIAL REPORT — Item {item.id_}  (phase={phase})")
        lines.append(sep)
        lines.append(f"Question:      {item.question}")
        lines.append(f"Ground Truth:  {item.answer}")
        lines.append(f"Elapsed:       {elapsed:.2f}s")
        if error:
            lines.append(f"Error:         {error}")
        lines.append("")

        # Clusters
        if self.builder.table_understanding:
            lines.append(sep)
            lines.append("STEP 0 — TABLE UNDERSTANDING")
            lines.append(sep)
            for table in self.builder.table_understanding.tables:
                lines.append(f"\n  Table: `{table.table_name}`  —  {table.description}")
                for column in table.columns:
                    lines.append(
                        f"    {column.column_name:<25s}  "
                        f"type={column.semantic_type:<16s}  "
                        f"desc={column.description}"
                    )

        if self.builder.clusterings:
            lines.append(sep)
            lines.append("STEP 1 — COLUMN CLUSTERS")
            lines.append(sep)
            for tname, cres in self.builder.clusterings.items():
                lines.append(f"\n  Table: `{tname}`  —  {cres.table_description}")
                for cl in cres.clusters:
                    lines.append(
                        f"    [{cl.cluster_id}]  {cl.label:<25s}  "
                        f"pk={cl.primary_key_candidate:<20s}  "
                        f"cols={cl.columns}"
                    )
                for prop in cres.relation_properties:
                    lines.append(
                        f"    [relprop] {prop.column_name:<18s} "
                        f"depends_on={prop.depends_on}  desc={prop.description}"
                    )

        # Merged classes
        if self.builder.merged_classes:
            lines.append("")
            lines.append(sep)
            lines.append("STEP 2 — MERGED CLASSES")
            lines.append(sep)
            for cls in self.builder.merged_classes.classes:
                lines.append(
                    f"  [{cls.class_id}]  {cls.label:<25s}  "
                    f"pk={cls.primary_key:<20s}  "
                    f"cols={cls.all_columns}"
                )

        # Relations
        if self.builder.relations:
            lines.append("")
            lines.append(sep)
            lines.append("STEP 3 — RELATIONS")
            lines.append(sep)
            for rel in self.builder.relations.relations:
                lines.append(
                    f"  {rel.source_class} --[{rel.relation}]--> "
                    f"{rel.target_class}  ({rel.cardinality})  "
                    f"props={[prop.name for prop in rel.properties]}"
                )

        # Raw table data
        lines.append("")
        lines.append(sep)
        lines.append("TABLE DATA")
        lines.append(sep)
        for tname, table in item.tables.items():
            lines.append(f"\n  [{tname}]  columns: {table.columns}")
            for row in table.rows:
                lines.append(f"    {row}")
        lines.append("")
        lines.append(sep)
        lines.append("END OF REPORT")
        lines.append(sep)

        log.debug("Partial verbose report built (%d lines); not written to disk", len(lines))

    def _save_instance_kg(self, instance_kg: InstanceKG, item_dir: Path) -> None:
        """Write the materialized instance KG to its own JSON file."""
        (item_dir / "instance_kg.json").write_text(
            json.dumps(instance_kg.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _report_stage3_instance(
        self, item: QAItem, instance_kg: InstanceKG, item_dir: Path
    ) -> None:
        """Log Stage-3 instance-KG output to console only."""
        header = f"=== STAGE 3: Instance KG (item {item.id_}) ==="
        num_entities = sum(len(instances) for instances in instance_kg.entities.values())
        body = (
            f"**Question**:     {item.question}\n"
            f"**Entity Types**:  {len(instance_kg.entities)}\n"
            f"**Entities**:      {num_entities}\n"
            f"**Triples**:       {len(instance_kg.triples)}"
        )
        log.info("%s\n%s\n%s", header, body, "=" * 50)

    def _save_cypher_pipeline(
        self, result: CypherRunResult, item_dir: Path
    ) -> None:
        """Write the two-step Cypher path artifacts to cypher_pipeline.json."""
        payload = {
            "item_id": result.item_id,
            "question": result.question,
            "answer": result.answer,
            "error": result.error,
            "decomposition": result.decomposition,
            "sub_questions": result.sub_questions,
            "queries": result.queries,
            "attempts": result.attempts,
            "issues": result.issues,
        }
        (item_dir / "cypher_pipeline.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _report_cypher(
        self, item: QAItem, result: CypherRunResult, item_dir: Path
    ) -> None:
        """Log Cypher-path output to console only."""
        header = f"=== STAGE 4: Two-step Cypher Pipeline (item {item.id_}) ==="
        body = (
            f"**Question**:      {item.question}\n"
            f"**Ground Truth**:  {item.answer}\n"
            f"**Predicted**:     {result.answer}\n"
            f"**Queries**:       {result.queries}\n"
            f"**Issues**:        {result.issues}\n"
            f"**Error**:         {result.error}"
        )
        log.info("%s\n%s\n%s", header, body, "=" * 50)

    def _save_direct_pipeline(self, result: DirectRunResult, item_dir: Path) -> None:
        """Write the direct-translation artifacts to direct_pipeline.json."""

        def _json_safe(value):
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            return value

        payload = {
            "item_id": result.item_id,
            "question": result.question,
            "answer": result.answer,
            "error": result.error,
            "cypher": result.cypher,
            "rows": _json_safe(result.rows),
            "schema_prompt": result.schema_prompt,
        }
        (item_dir / "direct_pipeline.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _report_direct(
        self, item: QAItem, result: DirectRunResult, item_dir: Path
    ) -> None:
        """Log direct Cypher translation output to console only."""
        header = f"=== STAGE 4: Direct Cypher Translation (item {item.id_}) ==="
        body = (
            f"**Question**:      {item.question}\n"
            f"**Ground Truth**:  {item.answer}\n"
            f"**Predicted**:     {result.answer}\n"
            f"**Cypher**:        {result.cypher}\n"
            f"**Error**:         {result.error}"
        )
        log.info("%s\n%s\n%s", header, body, "=" * 50)

    def _save_result_summary(
        self,
        item: QAItem,
        predicted: str,
        correct: bool,
        elapsed: float,
        error: str | None,
        item_dir: Path,
        source_item_id: int | None = None,
        source_item_index: int | None = None,
        question_index: int | None = None,
        output_item_id: int | None = None,
        extra: dict | None = None,
    ) -> None:
        """Save a compact JSON summary for one item."""
        summary = {
            "item_id": output_item_id if output_item_id is not None else item.id_,
            "question": item.question,
            "ground_truth": item.answer,
            "predicted": predicted,
            "correct": correct,
            "elapsed_sec": round(elapsed, 2),
            "error": error,
            "source_item_id": source_item_id,
            "source_item_index": source_item_index,
            "question_index": question_index,
        }
        if extra:
            summary.update(extra)
        (item_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_intermediate_json(
        self,
        item: QAItem,
        schema: GraphSchema | None,
        instance_kg: InstanceKG | None,
        predicted: str,
        correct: bool,
        elapsed: float,
        error: str | None,
        item_dir: Path,
        source_item_id: int | None = None,
        source_item_index: int | None = None,
        question_index: int | None = None,
        output_item_id: int | None = None,
    ) -> None:
        """Save one consolidated intermediate artifact for a single item run."""
        item_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "item": {
                "item_id": output_item_id if output_item_id is not None else item.id_,
                "question": item.question,
                "sql": item.sql,
                # "answer": item.answer,
                # "table_names": item.table_names,
            },
            # "input_schema": self._build_input_schema(item),
            "steps": {
                "table_understanding": (
                    self.builder.table_understanding.to_dict()
                    if self.builder.table_understanding else None
                ),
                "cluster": self._build_cluster_step(item),
                "merge": self.builder.merged_classes.reasoning if self.builder.merged_classes else None,
                # "relation": self.builder.relations.to_dict() if self.builder.relations else None,
                "schema": schema.to_dict() if schema else None,
                "instance": instance_kg.to_dict() if instance_kg else None,
            },
            "result": {
                "predicted": predicted,
                "golden answer": item.answer,
                "correct": correct,
                "elapsed_sec": round(elapsed, 2),
                "error": error,
                "source_item_id": source_item_id,
                "source_item_index": source_item_index,
                "question_index": question_index,
            },
        }
        (item_dir / "intermediate.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _resolve_output_item_id(item: QAItem, qa_pair: QAExample | None = None) -> int:
        """Pick the source item id to use for per-QA output naming and metadata."""
        if qa_pair is not None:
            if qa_pair.old_item_id is not None:
                return qa_pair.old_item_id
            if qa_pair.source_item_id is not None:
                return qa_pair.source_item_id
        return item.id_

    @staticmethod
    def _format_qa_folder_suffix(item_id: int | None) -> str:
        """Render a stable qa-folder suffix from the source item id."""
        if item_id is None:
            return "00"
        return str(int(item_id)).zfill(2)

    @staticmethod
    def _build_input_schema(item: QAItem) -> dict:
        """Project the dataset's table metadata into a per-table schema view."""
        primary_keys_by_table: dict[str, list[str]] = {name: [] for name in item.table_names}
        if len(item.primary_keys) == len(item.table_names):
            for table_name, primary_key in zip(item.table_names, item.primary_keys):
                if primary_key in item.tables[table_name].columns:
                    primary_keys_by_table[table_name].append(primary_key)
        else:
            pk_lookup = {key.lower() for key in item.primary_keys}
            for table_name in item.table_names:
                table = item.tables[table_name]
                primary_keys_by_table[table_name] = [
                    col for col in table.columns if col.lower() in pk_lookup
                ]

        fk_lookup = {key.lower() for key in item.foreign_keys}

        tables = []
        for table_name in item.table_names:
            table = item.tables[table_name]
            tables.append(
                {
                    "table_name": table_name,
                    "columns": table.columns,
                    "primary_keys_from_input": primary_keys_by_table[table_name],
                    "foreign_keys_matching_input_names": [
                        col for col in table.columns if col.lower() in fk_lookup
                    ],
                    "num_rows": table.num_rows,
                }
            )

        return {
            # "primary_keys_raw": item.primary_keys,
            # "foreign_keys_raw": item.foreign_keys,
            "tables": tables,
        }

    def _build_cluster_step(self, item: QAItem) -> dict | None:
        """Build cluster-step JSON, including coverage diagnostics per table."""
        if not self.builder.clusterings:
            return None

        tables = []
        for table_name in item.table_names:
            table = item.tables[table_name]
            clustering = self.builder.clusterings.get(table_name)
            if clustering is None:
                continue

            tables.append(
                {
                    "table_name": table_name,
                    # "table_description": clustering.table_description,
                    # "column_understanding": [
                    #     {
                    #         "column_name": column.column_name,
                    #         "description": column.description,
                    #         "semantic_type": column.semantic_type,
                    #     }
                    #     for column in (self.builder.table_understanding.get_table(table_name).columns
                    #                    if self.builder.table_understanding and self.builder.table_understanding.get_table(table_name)
                    #                    else [])
                    # ],
                    "clusters": [
                        {
                            "cluster_id": cluster.cluster_id,
                            "label": cluster.label,
                            "description": cluster.description,
                            "columns": cluster.columns,
                            "identify_column": cluster.primary_key_candidate,
                        }
                        for cluster in clustering.clusters
                    ],
                    "relation_properties": [
                        {
                            "column_name": prop.column_name,
                            "description": prop.description,
                            "depends_on": prop.depends_on,
                        }
                        for prop in clustering.relation_properties
                    ],
                    "missing_columns": clustering.missing_columns(table.columns),
                    "duplicate_columns": clustering.duplicate_columns(),
                }
            )

        return {"tables": tables}

    # ------------------------------------------------------------------
    # Run a dataset
    # ------------------------------------------------------------------
    def run_dataset(
        self,
        json_path: str | Path,
        max_items: int = -1,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> PipelineReport:
        """
        Run the pipeline on all items in a dataset JSON.

        Args:
            json_path: Path to the MMQA JSON file.
            max_items: Max items to process (-1 = all).
            save_intermediate: Whether to save graphs & plans to disk.

        Returns:
            PipelineReport with aggregate statistics.
        """
        items = load_dataset(json_path, max_items=max_items)
        log.info("Loaded %d items from %s", len(items), json_path)

        report = PipelineReport(total=len(items))
        t_start = time.perf_counter()

        for i, item in enumerate(items):
            log.info("=== Item %d/%d (id=%d) ===", i + 1, len(items), item.id_)
            result = self.run_one(item, verbose=verbose)
            report.results.append(result)

            if result.correct:
                report.correct += 1

            # Save intermediates
            if save_intermediate:
                self._save_result(result)

            # Progress
            acc_so_far = report.correct / (i + 1)
            log.info("  Result: %s | Acc so far: %.2f%% | Time: %.1fs",
                     "✓" if result.correct else "✗",
                     acc_so_far * 100,
                     result.elapsed_sec)

        report.total_time_sec = time.perf_counter() - t_start
        report.accuracy = report.correct / max(report.total, 1)

        # Save report
        self._save_report(report)

        log.info("=" * 50)
        log.info("PIPELINE COMPLETE")
        log.info("  Total: %d  Correct: %d  Accuracy: %.2f%%  Time: %.1fs",
                 report.total, report.correct,
                 report.accuracy * 100, report.total_time_sec)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_answer(predicted: str, ground_truth: str) -> bool:
        """Order-insensitive set comparison for list answers.

        Answers may be Python list/tuple literals (``"['a', 'b']"``), lists
        of lists, or loose values separated by ``|``, ``;`` or ``,``. The two
        answers are reduced to sets, so element order does not matter and a
        correct answer only needs the same elements.

        ``"(no result)"`` is considered correct when the ground truth is empty.

        When both answers are numeric, they are compared with a small
        tolerance to absorb floating-point drift from computation.
        """
        pred_norm = (predicted or "").strip().lower()
        gt_norm = (ground_truth or "").strip()
        if pred_norm == "(no result)" and gt_norm == "":
            return True

        pred_float = PipelineOrchestrator._to_float(predicted)
        gt_float = PipelineOrchestrator._to_float(ground_truth)
        if pred_float is not None and gt_float is not None:
            return math.isclose(pred_float, gt_float, rel_tol=1e-6, abs_tol=1e-9)

        if _answer_set(predicted) == _answer_set(ground_truth):
            return True
        # Format asymmetry: a structured (list-literal) ground truth vs the
        # executor's flat ", "-joined predicted answer. The executor joins list
        # cells with ", ", so a value containing a comma ("2,000 Man") is
        # fragmented in the predicted string but preserved whole in the literal.
        # Re-tokenise both sides at the same lossy granularity and re-compare.
        if _is_literal(predicted) != _is_literal(ground_truth):
            return _answer_set_flat(predicted) == _answer_set_flat(ground_truth)
        return False

    @staticmethod
    def _to_float(s: str) -> float | None:
        """Parse a string as a float, or return None if it is not numeric."""
        try:
            return float((s or "").strip())
        except (TypeError, ValueError):
            return None

    def _save_result(self, result: StageResult) -> None:
        """Save intermediate outputs for one item."""
        item_dir = self.output_dir / f"item_{result.item_id:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        # Graph schema
        if result.graph_schema:
            (item_dir / "graph_schema.json").write_text(
                json.dumps(result.graph_schema.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Summary
        summary = {
            "item_id": result.item_id,
            "question": result.question,
            "ground_truth": result.ground_truth,
            "predicted": result.predicted,
            "correct": result.correct,
            "elapsed_sec": round(result.elapsed_sec, 2),
            "error": result.error,
        }
        (item_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_report(self, report: PipelineReport) -> None:
        """Save the aggregate report."""
        report_path = self.output_dir / "report.json"
        data = report.to_dict()
        data["results"] = [
            {
                "item_id": r.item_id,
                "question": r.question[:120],
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "correct": r.correct,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "error": r.error,
            }
            for r in report.results
        ]
        report_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Report saved to %s", report_path)
