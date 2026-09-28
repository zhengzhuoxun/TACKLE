"""
Instance-Level Knowledge Graph Builder.

Given the GraphSchema (from Stage 1) and raw table data, materialise
the schema into instance-level entities and relation triples.

This is a separate module because instance-level population is
conceptually distinct from schema discovery and can be toggled
independently (e.g., for large tables, you may want schema only).
"""

from __future__ import annotations

from src.core.cluster_merger import MergedEntities
from src.core.column_clusterer import ClusteringResult
from src.core.table_understander import TableUnderstandingResult
from src.data.loader import QAItem
from src.core.graph_schema_builder import GraphSchema
from src.core.relation_generator import RelationResult
from src.core.instance_kg_builder import (
    InstanceKGBuilder as CoreInstanceKGBuilder,
    InstanceKG,
)
from src.llm.client import LLMClient, get_llm
from src.utils.logging import log


class InstanceGraphBuilder:
    """Builds the instance-level knowledge graph from schema + data."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()
        self._builder = CoreInstanceKGBuilder()

    def build(
        self,
        schema: GraphSchema,
        item: QAItem,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        relations: RelationResult,
        understanding: TableUnderstandingResult | None = None,
    ) -> InstanceKG:
        """
        Step 5: Materialise the graph schema into instance-level KG.

        Args:
            schema: GraphSchema from Stage 1 (steps 1-4).
            item: QA item with raw table content.
            clusterings: Per-table clustering outputs from Step 1.
            merged: Canonical entity mappings from Step 2.
            relations: Validated relations from Step 3.
            understanding: Step-0 table understanding (FK declarations).

        Returns:
            InstanceKG populated with entities and triples.
        """
        log.info("Stage 3 | Building instance-level KG for item %d", item.id_)
        kg = self._builder.build(schema, item, clusterings, merged, relations, understanding)
        log.info("Stage 3 | Instance KG: %d entity types, %d triples",
                 len(kg.entities), len(kg.triples))
        return kg
