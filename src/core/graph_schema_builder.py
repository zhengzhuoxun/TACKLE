"""
Step 4: Graph Schema Builder

Assembles the final graph schema deterministically from:
  - per-table clusters
  - merged entities
  - validated relations

This step is intentionally a wrapper/assembler rather than another LLM call.
"""

from __future__ import annotations

from pydantic import Field

from src.core.cluster_merger import MergedEntities, MergedEntity, EntityProperty
from src.core.column_clusterer import ClusteringResult
from src.core.relation_generator import (
    RelationResult,
    Relation,
    RelationPropertyAssignment,
)
from src.core.table_understander import TableUnderstandingResult, ColumnUnderstanding
from src.utils.schema import ProjectModel
from src.utils.dedup import deduplicate
from src.utils.logging import log


class Attribute(ProjectModel):
    """One attribute (property) of a graph node."""

    name: str
    semantic_type: str
    description: str
    sample_values: list[object] = Field(default_factory=list)


class SchemaNode(ProjectModel):
    """One node (entity class) in the graph schema."""

    node_id: str
    label: str
    description: str
    attributes: list[Attribute] = Field(default_factory=list)
    primary_key: str | list[str] = ""


class SchemaEdge(ProjectModel):
    """One directed edge in the graph schema."""

    source: str
    target: str
    label: str
    cardinality: str
    description: str
    properties: list[Attribute] = Field(default_factory=list)


class GraphSchema(ProjectModel):
    """The final graph schema — blueprint for the knowledge graph."""

    schema_name: str
    nodes: list[SchemaNode] = Field(default_factory=list)
    edges: list[SchemaEdge] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            # "schema_name": self.schema_name,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "type": "entity",
                    "description": n.description,
                    "primary_key": n.primary_key,
                    "attributes": [
                        {
                            "name": a.name,
                            "semantic_type": a.semantic_type,
                            "description": a.description,
                            "sample_values": a.sample_values,
                        }
                        for a in n.attributes
                    ],
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "label": e.label,
                    "cardinality": e.cardinality,
                    "description": e.description,
                    "properties": [
                        {
                            "name": a.name,
                            "semantic_type": a.semantic_type,
                            "description": a.description,
                            "sample_values": a.sample_values,
                        }
                        for a in e.properties
                    ],
                }
                for e in self.edges
            ],
        }

    def to_prompt_text(self) -> str:
        """Render a compact schema summary for downstream prompt use."""
        lines = [f"## Graph Schema: {self.schema_name}"]
        lines.append("\n### Nodes")
        for node in self.nodes:
            attrs = ", ".join(
                self._format_attribute(attr) for attr in node.attributes
            )
            lines.append(f"  - **{node.node_id}** ({node.label}): {node.description}")
            lines.append(f"    pk={node.primary_key}, attrs=[{attrs}]")
        lines.append("\n### Edges")
        for edge in self.edges:
            props = ", ".join(
                self._format_attribute(prop) for prop in edge.properties
            )
            lines.append(
                f"  - {edge.source} --[{edge.label}]--> {edge.target} "
                f"({edge.cardinality}): {edge.description}"
            )
            if props:
                lines.append(f"    edge_props=[{props}]")
        return "\n".join(lines)

    @staticmethod
    def _format_attribute(attr: Attribute) -> str:
        """Render attributes with sample values when available."""
        base = f"{attr.name} ({attr.semantic_type})"
        if attr.sample_values:
            return f"{base}[samples={attr.sample_values}]"
        return base


class GraphSchemaBuilder:
    """Assembles a clear, concise graph schema from entities and relations."""

    def build(
        self,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        relations: RelationResult,
        understanding: TableUnderstandingResult,
    ) -> GraphSchema:
        """Build the graph schema deterministically from prior stage outputs."""
        log.info(
            "  [Schema] Assembling graph schema from %d classes + %d relations",
            len(merged.classes),
            len(relations.relations),
        )

        # Build nodes from canonical merged entities so the final schema stays
        # stable and concise even when multiple raw clusters contributed to it.
        nodes = self._build_nodes_from_entities(merged, understanding)

        # Build edges directly from validated relations instead of asking the
        # LLM to reinterpret semantics that were already decided in Step 3.
        edges = self._build_edges_from_relations(relations, understanding)
        edges = self._dedupe_edges(edges)

        schema = GraphSchema(
            schema_name=self._build_schema_name(clusterings, merged),
            nodes=nodes,
            edges=edges,
        )
        self._validate_schema(schema)

        log.info("  [Schema] → %d nodes, %d edges", len(schema.nodes), len(schema.edges))
        return schema

    def _build_schema_name(
        self,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
    ) -> str:
        """Create a short deterministic schema name from the current data shape."""
        if len(merged.classes) == 1:
            return f"{merged.classes[0].class_id}_graph"
        if len(clusterings) == 1:
            table_name = next(iter(clusterings))
            return f"{table_name}_graph"
        return "enterprise_graph"

    def _build_nodes_from_entities(
        self,
        merged: MergedEntities,
        understanding: TableUnderstandingResult,
    ) -> list[SchemaNode]:
        """Convert merged entities into schema nodes with concise attributes."""
        nodes: list[SchemaNode] = []
        for entity in merged.classes:
            attributes = self._build_attributes_from_entity(entity, understanding)
            # Put primary key first for consistent downstream reading
            attributes = self._order_attributes_pk_first(attributes, entity.primary_key)
            node = SchemaNode(
                node_id=entity.class_id,
                label=entity.label or entity.class_id.replace("_", " ").title(),
                description=self._normalize_entity_description(entity),
                primary_key=entity.primary_key,
                attributes=attributes,
            )
            nodes.append(node)
        return nodes

    @staticmethod
    def _order_attributes_pk_first(
        attributes: list[Attribute],
        primary_key: str | list[str],
    ) -> list[Attribute]:
        """Move primary-key attribute(s) to the front of the list."""
        if not primary_key:
            return attributes
        # For composite keys (list), move all PK columns to the front.
        if isinstance(primary_key, list):
            pk_lower = {pk.lower() for pk in primary_key}
            result: list[Attribute] = []
            rest: list[Attribute] = []
            for attr in attributes:
                if attr.name.lower() in pk_lower:
                    result.append(attr)
                else:
                    rest.append(attr)
            return result + rest
        # Single-column PK.
        pk_lower = primary_key.lower()
        result: list[Attribute] = []
        rest: list[Attribute] = []
        for attr in attributes:
            if attr.name.lower() == pk_lower:
                result.append(attr)
            else:
                rest.append(attr)
        return result + rest

    def _build_attributes_from_entity(
        self,
        entity: MergedEntity,
        understanding: TableUnderstandingResult,
    ) -> list[Attribute]:
        """Map canonical entity properties into concise schema attributes."""
        attributes: list[Attribute] = []
        seen_names: set[str] = set()
        for prop in entity.properties:
            attr_name = prop.name.strip()
            if not attr_name:
                continue
            key = attr_name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            attributes.append(
                Attribute(
                    name=attr_name,
                    semantic_type=self._resolve_property_semantic_type(entity, prop, understanding),
                    description=self._resolve_property_description(entity, prop, understanding),
                    sample_values=self._resolve_property_sample_values(entity, prop, understanding),
                )
            )
        return attributes

    def _normalize_entity_description(self, entity: MergedEntity) -> str:
        """Keep entity descriptions short and readable in the final schema."""
        description = " ".join(entity.description.split()).strip()
        if description:
            return description
        label = entity.label or entity.class_id.replace("_", " ")
        return f"Represents {label.lower()}."

    def _resolve_property_description(
        self,
        entity: MergedEntity,
        prop: EntityProperty,
        understanding: TableUnderstandingResult,
    ) -> str:
        """Resolve property descriptions from Step-0 column understanding."""
        profiles = self._lookup_column_profiles(entity, prop, understanding)
        for profile in profiles:
            description = " ".join(profile.description.split()).strip()
            if description:
                return description
        description = " ".join(prop.description.split()).strip()
        if description:
            return description
        if prop.source_columns:
            return f"Derived from: {', '.join(prop.source_columns)}."
        return "Canonical entity property."

    def _resolve_property_semantic_type(
        self,
        entity: MergedEntity,
        prop: EntityProperty,
        understanding: TableUnderstandingResult,
    ) -> str:
        """Resolve property semantic types from Step-0 column understanding."""
        profiles = self._lookup_column_profiles(entity, prop, understanding)
        for profile in profiles:
            semantic_type = (profile.semantic_type or "").strip()
            if semantic_type:
                return semantic_type
        return "categorical"

    def _resolve_property_sample_values(
        self,
        entity: MergedEntity,
        prop: EntityProperty,
        understanding: TableUnderstandingResult,
    ) -> list[object]:
        """Resolve representative sample values for one entity property."""
        profiles = self._lookup_column_profiles(entity, prop, understanding)
        best_values: list[object] = []
        for profile in profiles:
            if len(profile.sample_values) > len(best_values):
                best_values = list(profile.sample_values[:3])
        if best_values:
            return best_values
        return list(prop.sample_values[:3])

    def _lookup_column_profiles(
        self,
        entity: MergedEntity,
        prop: EntityProperty,
        understanding: TableUnderstandingResult,
    ) -> list[ColumnUnderstanding]:
        """Find Step-0 column profiles supporting one canonical property."""
        source_tables = {
            cluster_ref.split(".", 1)[0]
            for cluster_ref in entity.source_clusters
            if "." in cluster_ref
        }
        matches: list[tuple[str, ColumnUnderstanding]] = []
        seen: set[tuple[str, str]] = set()
        for table_name in source_tables:
            table = understanding.get_table(table_name)
            if table is None:
                continue
            for column_name in prop.source_columns:
                profile = table.get_column(column_name)
                if profile is None:
                    continue
                key = (table_name.lower(), profile.column_name.lower())
                if key in seen:
                    continue
                seen.add(key)
                matches.append((table_name, profile))
        return self._order_profiles_by_relevance(matches, prop.name)

    @staticmethod
    def _order_profiles_by_relevance(
        profiles: list[tuple[str, ColumnUnderstanding]],
        prop_name: str,
    ) -> list[ColumnUnderstanding]:
        """Sort profiles so the most relevant one (table + column name match) comes first."""
        prop_lower = prop_name.lower()

        def _score(item: tuple[str, ColumnUnderstanding]) -> tuple[int, int]:
            table_name, profile = item
            col_lower = profile.column_name.lower()
            tbl_lower = table_name.lower()
            # Primary: exact column name match (e.g. "balance" == "balance")
            col_score = 0
            if col_lower == prop_lower:
                col_score = 2
            elif prop_lower in col_lower or col_lower in prop_lower:
                col_score = 1
            # Secondary: table name appears in property name
            # (e.g. "checking" in "checking_balance" → prefer CHECKING.balance over SAVINGS.balance)
            tbl_score = 1 if tbl_lower in prop_lower else 0
            return (col_score, tbl_score)

        return [
            profile
            for _, profile in sorted(profiles, key=_score, reverse=True)
        ]

    def _build_edges_from_relations(
        self,
        relations: RelationResult,
        understanding: TableUnderstandingResult,
    ) -> list[SchemaEdge]:
        """Convert validated relations directly into schema edges."""
        edges: list[SchemaEdge] = []
        for relation in relations.relations:
            edges.append(
                SchemaEdge(
                    source=relation.source_class,
                    target=relation.target_class,
                    label=relation.relation,
                    cardinality=relation.cardinality,
                    description=self._normalize_edge_description(relation),
                    properties=self._build_edge_properties(relation, understanding),
                )
            )
        return edges

    def _build_edge_properties(
        self,
        relation: Relation,
        understanding: TableUnderstandingResult,
    ) -> list[Attribute]:
        """Map relation properties into concise schema edge attributes."""
        attributes: list[Attribute] = []
        seen_names: set[str] = set()
        for prop in relation.properties:
            attr_name = prop.name.strip()
            if not attr_name:
                continue
            key = attr_name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            attributes.append(
                Attribute(
                    name=attr_name,
                    semantic_type=self._resolve_relation_property_semantic_type(
                        relation,
                        prop,
                        understanding,
                    ),
                    description=self._resolve_relation_property_description(
                        relation,
                        prop,
                        understanding,
                    ),
                    sample_values=self._resolve_relation_property_sample_values(
                        relation,
                        prop,
                        understanding,
                    ),
                )
            )
        return attributes

    def _normalize_edge_description(self, relation: Relation) -> str:
        """Keep edge descriptions short and deterministic."""
        description = " ".join(relation.description.split()).strip()
        if description:
            return description
        return (
            f"{relation.source_class} {relation.relation.replace('_', ' ')} "
            f"{relation.target_class}."
        )

    def _dedupe_edges(self, edges: list[SchemaEdge]) -> list[SchemaEdge]:
        """Collapse exact duplicate edges to keep the final schema concise."""
        return deduplicate(
            edges,
            key=lambda e: (e.source, e.target, e.label, e.cardinality),
        )

    def _resolve_relation_property_description(
        self,
        relation: Relation,
        prop: RelationPropertyAssignment,
        understanding: TableUnderstandingResult,
    ) -> str:
        """Resolve edge-property descriptions from Step-0 column understanding."""
        profile = self._lookup_relation_property_profile(relation, prop, understanding)
        if profile is not None:
            description = " ".join(profile.description.split()).strip()
            if description:
                return description
        description = " ".join(prop.description.split()).strip()
        if description:
            return description
        return f"Derived from relation column `{prop.source_column}`."

    def _resolve_relation_property_semantic_type(
        self,
        relation: Relation,
        prop: RelationPropertyAssignment,
        understanding: TableUnderstandingResult,
    ) -> str:
        """Resolve edge-property semantic types from Step-0 column understanding."""
        profile = self._lookup_relation_property_profile(relation, prop, understanding)
        if profile is not None:
            semantic_type = (profile.semantic_type or "").strip()
            if semantic_type:
                return semantic_type
        return "categorical"

    def _resolve_relation_property_sample_values(
        self,
        relation: Relation,
        prop: RelationPropertyAssignment,
        understanding: TableUnderstandingResult,
    ) -> list[object]:
        """Resolve representative sample values for one edge property."""
        profile = self._lookup_relation_property_profile(relation, prop, understanding)
        if profile is not None and profile.sample_values:
            return _dedupe_preserve_order_objects(profile.sample_values)
        return _dedupe_preserve_order_objects(prop.sample_values)

    def _lookup_relation_property_profile(
        self,
        relation: Relation,
        prop: RelationPropertyAssignment,
        understanding: TableUnderstandingResult,
    ) -> ColumnUnderstanding | None:
        """Find the supporting Step-0 column profile for a relation property."""
        table_name = relation.source_cluster.split(".", 1)[0]
        table = understanding.get_table(table_name)
        if table is None:
            return None
        return table.get_column(prop.source_column)

    @staticmethod
    def _format_attribute(attr: Attribute) -> str:
        """Render attributes with sample values when available."""
        base = f"{attr.name} ({attr.semantic_type})"
        if attr.sample_values:
            return f"{base}[samples={attr.sample_values}]"
        return base

    def _validate_schema(self, schema: GraphSchema) -> None:
        """Fail fast if the assembled schema violates basic graph invariants."""
        node_ids = {node.node_id for node in schema.nodes}
        if any(not node.node_id for node in schema.nodes):
            raise ValueError("Schema contains a node with an empty node_id")

        for node in schema.nodes:
            attr_names = [attr.name.lower() for attr in node.attributes]
            if len(attr_names) != len(set(attr_names)):
                raise ValueError(f"Schema node '{node.node_id}' has duplicate attributes")

        for edge in schema.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"Schema edge '{edge.source} -> {edge.target}' references unknown nodes"
                )
            prop_names = [prop.name.lower() for prop in edge.properties]
            if len(prop_names) != len(set(prop_names)):
                raise ValueError(
                    f"Schema edge '{edge.source} -[{edge.label}]-> {edge.target}' has duplicate properties"
                )


def _dedupe_preserve_order_objects(values: list[object]) -> list[object]:
    seen: set[str] = set()
    deduped: list[object] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
