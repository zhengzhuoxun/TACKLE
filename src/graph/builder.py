"""
Stage 1: Semantic Graph Builder (Schema Level — Steps 0-4)

Orchestrates the five core steps that build a semantic graph schema
from noisy enterprise tables, WITHOUT relying on FK/PK metadata:

  Step 0 — TableUnderstander:  understand all tables/columns jointly
  Step 1 — ColumnClusterer:   cluster columns within each table
  Step 2 — ClassMerger:        merge equivalent clusters across tables
  Step 3 — RelationGenerator:  identify relations between clusters/classes
  Step 4 — GraphSchemaBuilder: assemble the final graph schema

Input:  QAItem (tables with columns and rows, NO FK/PK assumptions).
Output: GraphSchema (nodes and edges).
"""

from __future__ import annotations

from config.settings import KGQA_PHASES, _resolve_phase

from src.data.loader import QAItem
from src.core.column_clusterer import ColumnClusterer, ClusteringResult
from src.core.cluster_instance_extractor import ClusterInstanceExtractor, ClusterInstance
from src.core.cluster_merger import ClassMerger, MergedEntities
from src.core.relation_generator import RelationGenerator, RelationResult
from src.core.graph_schema_builder import GraphSchemaBuilder, GraphSchema
from src.core.table_understander import TableUnderstander, TableUnderstandingResult
from src.llm.client import LLMClient, get_llm
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SemanticGraphBuilder:
    """Builds a GraphSchema from table data using a 5-step pipeline.

    Does NOT use FK/PK metadata.  Treats every table as a flat, potentially
    de-normalised collection of columns that may embed multiple entities.
    """

    def __init__(self, llm: LLMClient | None = None):
        llm = llm or get_llm()
        # Step handlers
        self.understander = TableUnderstander(llm)
        self.clusterer = ColumnClusterer(llm)
        self.instance_extractor = ClusterInstanceExtractor()
        self.merger = ClassMerger(llm)
        self.relater = RelationGenerator(llm)
        self.schema_builder = GraphSchemaBuilder()

        # Intermediate state (accessible after build)
        self._table_understanding: TableUnderstandingResult | None = None
        self._clusterings: dict[str, ClusteringResult] | None = None
        self._cluster_instances: list[ClusterInstance] | None = None
        self._merged: MergedEntities | None = None
        self._relations: RelationResult | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    from typing import Any  # noqa: F811 — polymorphic return: type depends on phase

    def build(
        self, item: QAItem,
        phase: str | int = "schema",
    ) -> GraphSchema | MergedEntities | RelationResult | dict[str, ClusteringResult]:
        """
        Stage 1: Build the semantic graph schema from table data.

        Return type depends on *phase*:
          "cluster"  / 1 → dict[str, ClusteringResult]
          "merge"    / 2 → MergedEntities
          "relation" / 3 → RelationResult
          "schema"   / 4 → GraphSchema (default)

        Args:
            item: The QA item to process.
            phase: How many steps to run before stopping.
        """
        stop_at = _resolve_phase(phase)

        log.info("=" * 50)
        log.info("Stage 1 | Building semantic graph for item %d (%d tables)  phase=%s → step %d",
                 item.id_, item.num_tables, phase, stop_at)

        # ---- Step 0: Understand all tables jointly ----
        log.info("--- Step 0/4: Table Understanding ---")
        self._table_understanding = self.understander.understand(item)
        self._print_table_understanding_summary()

        # ---- Step 1: Cluster columns within each table ----
        log.info("--- Step 1/4: Column Clustering ---")
        self._clusterings = self.clusterer.cluster_all(item, self._table_understanding)
        self._print_cluster_summary()

        # ---- Step 1.5: Extract cluster instances & validate PK consistency ----
        log.info("--- Step 1.5/4: Cluster Instance Extraction ---")
        self._cluster_instances = self.instance_extractor.extract(item, self._clusterings)

        if stop_at <= 1:
            log.info("Stage 1 | Stopping after Step 1 Clustering") 
            return self._clusterings

        # ---- Step 2: Merge clusters across tables into classes ----
        log.info("--- Step 2/4: Class Merging ---")
        self._merged = self.merger.merge(self._clusterings, self._table_understanding)
        self._print_merge_summary()

        if stop_at <= 2:
            log.info("Stage 1 | Stopping after Step 2 Merging")
            return self._merged

        # ---- Step 3: Generate relations between clusters/classes ----
        log.info("--- Step 3/4: Relation Generation ---")
        self._relations = self.relater.generate(
            self._clusterings,
            self._merged,
            self._table_understanding,
        )
        self._print_relation_summary()

        if stop_at <= 3:
            log.info("Stage 1 | Stopping after Step 3 Relation Generation")
            return self._relations

        # ---- Step 4: Assemble the final graph schema ----
        log.info("--- Step 4/4: Graph Schema Output ---")
        schema = self.schema_builder.build(
            self._clusterings,
            self._merged,
            self._relations,
            self._table_understanding,
        )

        log.info("Stage 1 | Complete: %d nodes, %d edges",
                 len(schema.nodes), len(schema.edges))
        return schema

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _print_table_understanding_summary(self) -> None:
        """Print a compact summary of the shared Step-0 schema understanding."""
        if not self._table_understanding:
            return
        log.info("")
        log.info("--- Table Understanding ---")
        for table in self._table_understanding.tables:
            log.info("  Table: `%s`  (%s)", table.table_name, table.description)
            for column in table.columns:
                log.info(
                    "    %-20s  type=%-16s  samples=%s  desc=%s",
                    column.column_name,
                    column.semantic_type,
                    column.sample_values,
                    column.description,
                )

    def _print_cluster_summary(self) -> None:
        """Print a human-readable summary of all per-table clusters."""
        if not self._clusterings:
            return
        log.info("")
        log.info("--- Column Cluster Results ---")
        for tname, cres in self._clusterings.items():
            log.info("  Table: `%s`  (%s)", tname, cres.table_description)
            for cl in cres.clusters:
                cols_str = ", ".join(cl.columns)
                log.info("    [%s]  %-20s  pk=%-20s  cols=[%s]",
                         cl.cluster_id, cl.label, cl.primary_key_candidate, cols_str)
            for prop in cres.relation_properties:
                log.info(
                    "    [relprop] %-20s depends_on=%s desc=%s",
                    prop.column_name,
                    prop.depends_on,
                    prop.description,
                )

    def _print_merge_summary(self) -> None:
        """Print a human-readable summary of merged classes."""
        if not self._merged:
            return
        log.info("")
        log.info("--- Merged Classes ---")
        for cls in self._merged.classes:
            log.info("  [%s]  %-20s  pk=%-20s  cols=%s",
                     cls.class_id, cls.label, cls.primary_key, cls.all_columns)
        if self._merged.unmerged_clusters:
            log.info("  Unmerged: %s", self._merged.unmerged_clusters)

    def _print_relation_summary(self) -> None:
        """Print a human-readable summary of relations."""
        if not self._relations:
            return
        log.info("")
        log.info("--- Relations ---")
        for rel in self._relations.relations:
            prop_names = [prop.name for prop in rel.properties]
            log.info(
                "  %s --[%s]--> %s  (%s) props=%s",
                rel.source_class,
                rel.relation,
                rel.target_class,
                rel.cardinality,
                prop_names,
            )

    # ------------------------------------------------------------------
    # Accessors for intermediate outputs
    # ------------------------------------------------------------------
    @property
    def table_understanding(self) -> TableUnderstandingResult | None:
        """Holistic table and column understanding (Step 0 output)."""
        return self._table_understanding

    @property
    def clusterings(self) -> dict[str, ClusteringResult] | None:
        """Per-table column clusters (Step 1 output)."""
        return self._clusterings

    @property
    def cluster_instances(self) -> list[ClusterInstance] | None:
        """Pre-extracted cluster instances (Step 1.5 output)."""
        return self._cluster_instances

    @property
    def merged_classes(self) -> MergedEntities | None:
        """Merged classes across tables (Step 2 output)."""
        return self._merged

    @property
    def relations(self) -> RelationResult | None:
        """Inter-cluster/class relations (Step 3 output)."""
        return self._relations
