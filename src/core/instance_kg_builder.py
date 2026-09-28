"""
Step 5: Deterministic Instance-Level Knowledge Graph Builder

Materialises a graph schema into an instance KG without any LLM calls.

High-level flow:
  1. Split each raw table row into cluster-level row fragments.
  2. Map each cluster fragment to its merged entity class.
  3. Merge fragments with the same (class_id, identifier_value) into one entity.
  4. Create relation edges from row-local cluster co-occurrence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx 
# TODO: other instance graph backends like neo4j

from pydantic import Field

from src.core.cluster_instance_extractor import ClusterInstance
from src.core.cluster_merger import MergedEntities, MergedEntity
from src.core.column_clusterer import ClusteringResult, ColumnCluster
from src.core.table_understander import TableUnderstandingResult
from src.core.graph_schema_builder import GraphSchema
from src.core.relation_generator import RelationResult, Relation
from src.data.loader import QAItem, Table
from src.utils.logging import log
from src.utils.schema import ProjectModel



class EntityInstance(ProjectModel):
    """One merged entity instance in the instance KG."""

    class_id: str
    id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_fragments: list[dict[str, Any]] = Field(default_factory=list)


class Triple(ProjectModel):
    """One relation triple between two entity instances."""

    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = Field(default_factory=dict)
    table_name: str = ""
    row_index: int = -1
    source_cluster: str = ""
    target_cluster: str = ""

    @property
    def edge_id(self) -> str:
        return (
            f"{self.subject}|{self.predicate}|{self.object}|"
            f"{self.table_name}|{self.row_index}"
        )


class InstanceKG(ProjectModel):
    """Deterministic instance-level KG backed by a NetworkX MultiDiGraph."""

    graph_schema: GraphSchema
    entities: dict[str, list[EntityInstance]] = Field(default_factory=dict)
    triples: list[Triple] = Field(default_factory=list)
    cluster_instances: list[ClusterInstance] = Field(default_factory=list)
    graph: Any = None
    nodes_by_class: dict[str, list[str]] = Field(default_factory=dict)
    edge_index: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        graph = self.graph
        return {
            "graph_summary": {
                "num_nodes": graph.number_of_nodes() if graph is not None else 0,
                "num_edges": graph.number_of_edges() if graph is not None else 0,
                "num_triples": len(self.triples),
                "classes": sorted(self.entities.keys()),
                "relations": sorted({triple.predicate for triple in self.triples}),
            },
            "entities": {
                class_id: [
                    {
                        "id": entity.id,
                        "attributes": entity.attributes,
                        # "source_fragments": entity.source_fragments,
                    }
                    for entity in instances
                ]
                for class_id, instances in self.entities.items()
            },
            "triples": [triple.model_dump(mode="python") for triple in self.triples],
            # "cluster_instances": [
            #     fragment.model_dump(mode="python")
            #     for fragment in self.cluster_instances
            # ],
            # "nodes_by_class": self.nodes_by_class,
        }

    def query_entities(self, class_id: str) -> list[EntityInstance]:
        """Return all entity instances for one schema class."""
        return self.entities.get(class_id, [])

    def query_relations(self, subject_id: str, predicate: str | None = None) -> list[Triple]:
        """Return relation triples from one subject, optionally filtered by predicate."""
        if predicate is None:
            return [triple for triple in self.triples if triple.subject == subject_id]
        return [
            triple for triple in self.triples
            if triple.subject == subject_id and triple.predicate == predicate
        ]

    def materialized_schema(self) -> GraphSchema:
        """Return the graph schema pruned to relations that actually materialised.

        The schema is assembled before table rows are turned into triples, so
        it can advertise a relation no row ever supports (e.g. a value-overlap
        "FK" whose column is not the child cluster's identifier). Downstream
        planning must only traverse relations the instance KG really contains,
        otherwise the generated Cypher references a relation table Kuzu never
        created.
        """
        materialized = {triple.predicate for triple in self.triples}
        pruned_edges = [
            edge for edge in self.graph_schema.edges if edge.label in materialized
        ]
        return self.graph_schema.model_copy(update={"edges": pruned_edges})


class InstanceKGBuilder:
    """Materialise a schema and table rows into an instance-level NetworkX graph."""

    def build(
        self,
        schema: GraphSchema,
        item: QAItem,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        relations: RelationResult,
        understanding: TableUnderstandingResult | None = None,
    ) -> InstanceKG:
        """Build the deterministic instance KG from Stage 1 outputs."""

        total_rows = sum(table.num_rows for table in item.tables.values())
        log.info(
            "  [InstKG] Building instance KG from %d tables, %d rows, %d classes, %d relations",
            len(item.tables),
            total_rows,
            len(merged.classes),
            len(relations.relations),
        )

        rekey_maps = self._build_fk_rekey_maps(
            item=item,
            clusterings=clusterings,
            merged=merged,
            understanding=understanding,
        )
        cluster_instances, row_cluster_map = self._materialize_cluster_instances(
            item=item,
            clusterings=clusterings,
            merged=merged,
            rekey_maps=rekey_maps,
        )
        entities, node_lookup = self._merge_entity_instances(
            schema=schema,
            cluster_instances=cluster_instances,
        )
        triples = self._materialize_relation_triples(
            row_cluster_map=row_cluster_map,
            relations=relations,
            node_lookup=node_lookup,
            cluster_instances=cluster_instances,
        )
        graph, nodes_by_class = self._build_graph(
            schema=schema,
            entities=entities,
            triples=triples,
        )
        edge_index = self._build_edge_index(triples)

        kg = InstanceKG(
            graph_schema=schema,
            entities=entities,
            triples=triples,
            cluster_instances=cluster_instances,
            graph=graph,
            nodes_by_class=nodes_by_class,
            edge_index=edge_index,
        )
        log.info(
            "  [InstKG] → %d entity instances, %d cluster fragments, %d triples",
            sum(len(instances) for instances in entities.values()),
            len(cluster_instances),
            len(triples),
        )
        return kg

    def _materialize_cluster_instances(
        self,
        item: QAItem,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        rekey_maps: dict[str, dict[str, str]] | None = None,
    ) -> tuple[list[ClusterInstance], dict[tuple[str, int], dict[str, ClusterInstance]]]:
        """Split each row into cluster-specific fragments with stable identifiers."""
        cluster_instances: list[ClusterInstance] = []
        row_cluster_map: dict[tuple[str, int], dict[str, ClusterInstance]] = {}
        skipped_fragments = 0
        rekey_maps = rekey_maps or {}

        for table_name in item.table_names:
            table = item.tables[table_name]
            clustering = clusterings.get(table_name)
            if clustering is None:
                raise ValueError(f"Missing clustering for table '{table_name}'")

            for row_index, row in enumerate(table.rows):
                cluster_map_for_row: dict[str, ClusterInstance] = {}
                row_values = {
                    col: row[idx]
                    for idx, col in enumerate(table.columns)
                    if idx < len(row)
                }

                for cluster in clustering.clusters:
                    entity = merged.find_class_for_cluster(table_name, cluster.cluster_id)
                    if entity is None:
                        raise ValueError(
                            f"Cluster '{table_name}.{cluster.cluster_id}' has no merged entity"
                        )

                    # Build composite identifier from ALL PK columns.
                    pk_cols = self._get_pk_columns(cluster)
                    identifier_parts: list[str] = []
                    missing_pk = False
                    for col in pk_cols:
                        val = row_values.get(col)
                        if self._is_missing(val):
                            missing_pk = True
                            break
                        identifier_parts.append(self._to_identifier(val))

                    if missing_pk or not identifier_parts:
                        skipped_fragments += 1
                        continue

                    identifier_column = pk_cols[0]
                    identifier_value = "|".join(identifier_parts)

                    cluster_ref = f"{table_name}.{cluster.cluster_id}"
                    rekey_map = rekey_maps.get(cluster_ref)
                    if rekey_map and identifier_value in rekey_map:
                        identifier_value = rekey_map[identifier_value]

                    fragment = ClusterInstance(
                        table_name=table_name,
                        row_index=row_index,
                        cluster_ref=cluster_ref,
                        class_id=entity.class_id,
                        identifier_column=identifier_column,
                        identifier_value=identifier_value,
                        properties=self._extract_cluster_properties(
                            row_values=row_values,
                            cluster=cluster,
                            entity=entity,
                            table_name=table_name,
                        ),
                        source_columns=list(cluster.columns),
                        row_values=dict(row_values),
                    )
                    cluster_instances.append(fragment)
                    cluster_map_for_row[cluster_ref] = fragment

                row_cluster_map[(table_name, row_index)] = cluster_map_for_row

        if skipped_fragments:
            log.info("  [InstKG] Skipped %d cluster fragments with missing identifiers", skipped_fragments)
        return cluster_instances, row_cluster_map

    def _build_fk_rekey_maps(
        self,
        item: QAItem,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        understanding: TableUnderstandingResult | None,
    ) -> dict[str, dict[str, str]]:
        """Map each FK-keyed cluster's rows to its entity's canonical key.

        When a cluster's identifier column is a declared FK that references a
        *secondary* key of its merged entity (e.g. ``appellation_name`` →
        ``terroirs.region`` while the entity is keyed by ``identifier``), the
        cluster's rows must be re-keyed through the referenced column so they
        unify with the entity's primary-keyed rows instead of spawning
        duplicate nodes.
        """
        rekey_maps: dict[str, dict[str, str]] = {}
        if understanding is None:
            return rekey_maps

        for table_name, clustering in clusterings.items():
            table_understanding = understanding.get_table(table_name)
            if table_understanding is None:
                continue
            fks = table_understanding.foreign_keys
            if not fks:
                continue
            for cluster in clustering.clusters:
                cluster_ref = f"{table_name}.{cluster.cluster_id}"
                entity = merged.find_class_for_cluster(table_name, cluster.cluster_id)
                if entity is None:
                    continue
                pk_cols = self._get_pk_columns(cluster)
                if len(pk_cols) != 1:
                    continue  # only re-key single-column FK identifiers
                id_col = pk_cols[0]
                fk = next(
                    (f for f in fks if f.column.lower() == id_col.lower()),
                    None,
                )
                if fk is None or not fk.references_column:
                    continue
                if fk.references_table.lower() == table_name.lower():
                    continue  # self-referencing FK not supported here
                ref_understanding = understanding.get_table(fk.references_table)
                if ref_understanding is None:
                    continue
                ref_col = ref_understanding.get_column(fk.references_column)
                if ref_col is None or ref_col.identifier_kind != "primary_key":
                    continue  # only re-key through identifier columns
                entity_pk = entity.primary_key
                if isinstance(entity_pk, list):
                    entity_pk = entity_pk[0] if entity_pk else ""
                if not entity_pk:
                    continue
                if entity_pk.lower() == fk.references_column.lower():
                    continue  # FK already references the primary key directly
                ref_table = item.tables.get(fk.references_table)
                if ref_table is None:
                    continue
                ref_idx = ref_table.column_index(fk.references_column)
                pk_idx = ref_table.column_index(entity_pk)
                if ref_idx is None or pk_idx is None:
                    continue
                rekey_map: dict[str, str] = {}
                for row in ref_table.rows:
                    ref_val = row[ref_idx] if ref_idx < len(row) else None
                    pk_val = row[pk_idx] if pk_idx < len(row) else None
                    if self._is_missing(ref_val) or self._is_missing(pk_val):
                        continue
                    rekey_map[self._to_identifier(ref_val)] = self._to_identifier(pk_val)
                if rekey_map:
                    rekey_maps[cluster_ref] = rekey_map
        return rekey_maps

    def _merge_entity_instances(
        self,
        schema: GraphSchema,
        cluster_instances: list[ClusterInstance],
    ) -> tuple[dict[str, list[EntityInstance]], dict[tuple[str, str], str]]:
        """Merge partial fragments with the same class/id into canonical entities."""
        grouped: dict[tuple[str, str], list[ClusterInstance]] = defaultdict(list)
        for fragment in cluster_instances:
            grouped[(fragment.class_id, fragment.identifier_value)].append(fragment)

        schema_nodes = {node.node_id: node for node in schema.nodes}
        entities: dict[str, list[EntityInstance]] = defaultdict(list)
        node_lookup: dict[tuple[str, str], str] = {}

        for (class_id, identifier_value), fragments in grouped.items():
            merged_properties: dict[str, Any] = {}
            source_fragments: list[dict[str, Any]] = []
            for fragment in fragments:
                source_fragments.append(
                    {
                        "table_name": fragment.table_name,
                        "row_index": fragment.row_index,
                        "cluster_ref": fragment.cluster_ref,
                        "identifier_column": fragment.identifier_column,
                    }
                )
                for key, value in fragment.properties.items():
                    if key not in merged_properties or self._is_missing(merged_properties[key]):
                        merged_properties[key] = value

            entity_instance = EntityInstance(
                class_id=class_id,
                id=identifier_value,
                attributes=merged_properties,
                source_fragments=source_fragments,
            )
            entities[class_id].append(entity_instance)
            node_lookup[(class_id, identifier_value)] = self._node_id(class_id, identifier_value)

        for class_id, instances in entities.items():
            primary_key = schema_nodes.get(class_id).primary_key if class_id in schema_nodes else ""
            instances.sort(key=lambda entity: self._sortable_entity_key(entity, primary_key))

        return dict(entities), node_lookup

    def _materialize_relation_triples(
        self,
        row_cluster_map: dict[tuple[str, int], dict[str, ClusterInstance]],
        relations: RelationResult,
        node_lookup: dict[tuple[str, str], str],
        cluster_instances: list[ClusterInstance],
    ) -> list[Triple]:
        """Create relation triples from row-local co-occurrence and FK joins."""
        same_table: list[Relation] = []
        cross_table: list[Relation] = []
        for relation in relations.relations:
            source_table = relation.source_cluster.split(".", 1)[0]
            target_table = relation.target_cluster.split(".", 1)[0]
            if source_table == target_table:
                same_table.append(relation)
            else:
                cross_table.append(relation)

        triples = self._materialize_same_table_triples(
            row_cluster_map=row_cluster_map,
            relations=same_table,
            node_lookup=node_lookup,
        )
        triples.extend(
            self._materialize_cross_table_triples(
                cluster_instances=cluster_instances,
                relations=cross_table,
                node_lookup=node_lookup,
            )
        )
        triples.sort(key=lambda triple: (triple.subject, triple.predicate, triple.object, triple.row_index))
        return self._dedupe_triples(triples)

    @staticmethod
    def _triple_dedup_key(triple: Triple) -> tuple:
        """Identity key for one relationship fact.

        Two triples are the same fact when they share subject, predicate,
        object AND identical concrete edge properties. Duplicate source rows
        (e.g. a machine listed once per product family) otherwise materialise
        as several identical parallel edges, which inflate downstream counts
        such as ``count(n0)``.
        """
        props = tuple(sorted((str(k), str(v)) for k, v in triple.properties.items()))
        return (triple.subject, triple.predicate, triple.object, props)

    @classmethod
    def _dedupe_triples(cls, triples: list[Triple]) -> list[Triple]:
        """Drop redundant identical edges, keeping the first occurrence.

        Edges with the SAME endpoints but DIFFERENT properties are preserved
        (e.g. two installs of the same part on different dates).
        """
        seen: set[tuple] = set()
        deduped: list[Triple] = []
        for triple in triples:
            key = cls._triple_dedup_key(triple)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(triple)
        return deduped

    def _materialize_same_table_triples(
        self,
        row_cluster_map: dict[tuple[str, int], dict[str, ClusterInstance]],
        relations: list[Relation],
        node_lookup: dict[tuple[str, str], str],
    ) -> list[Triple]:
        """Create relation triples from row-local cluster co-occurrence."""
        relations_by_table: dict[str, list[Relation]] = defaultdict(list)
        for relation in relations:
            table_name = relation.source_cluster.split(".", 1)[0]
            relations_by_table[table_name].append(relation)

        triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for (table_name, row_index), fragments in row_cluster_map.items():
            for relation in relations_by_table.get(table_name, []):
                source_fragment = fragments.get(relation.source_cluster)
                target_fragment = fragments.get(relation.target_cluster)
                if source_fragment is None or target_fragment is None:
                    continue

                subject = node_lookup.get(
                    (source_fragment.class_id, source_fragment.identifier_value)
                )
                obj = node_lookup.get(
                    (target_fragment.class_id, target_fragment.identifier_value)
                )
                if subject is None or obj is None:
                    continue

                key = (subject, relation.relation, obj, table_name, row_index)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        subject=subject,
                        predicate=relation.relation,
                        object=obj,
                        properties=self._extract_relation_properties(
                            relation=relation,
                            source_fragment=source_fragment,
                            target_fragment=target_fragment,
                            table_name=table_name,
                            row_index=row_index,
                        ),
                        table_name=table_name,
                        row_index=row_index,
                        source_cluster=relation.source_cluster,
                        target_cluster=relation.target_cluster,
                    )
                )

        return triples

    def _materialize_cross_table_triples(
        self,
        cluster_instances: list[ClusterInstance],
        relations: list[Relation],
        node_lookup: dict[tuple[str, str], str],
    ) -> list[Triple]:
        """Create triples for cross-table FK relations by joining on identifier.

        A FK relation (e.g. `customer.custid` → `savings_account.custid`) links
        fragments from different tables whose identifier values are equal.
        """
        if not relations:
            return []

        fragments_by_cluster: dict[str, list[ClusterInstance]] = defaultdict(list)
        for fragment in cluster_instances:
            fragments_by_cluster[fragment.cluster_ref].append(fragment)

        triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for relation in relations:
            target_by_id = {
                fragment.identifier_value: fragment
                for fragment in fragments_by_cluster.get(relation.target_cluster, [])
            }
            for source_fragment in fragments_by_cluster.get(relation.source_cluster, []):
                target_fragment = target_by_id.get(source_fragment.identifier_value)
                if target_fragment is None:
                    continue

                subject = node_lookup.get(
                    (source_fragment.class_id, source_fragment.identifier_value)
                )
                obj = node_lookup.get(
                    (target_fragment.class_id, target_fragment.identifier_value)
                )
                if subject is None or obj is None:
                    continue

                key = (
                    subject,
                    relation.relation,
                    obj,
                    relation.source_cluster,
                    source_fragment.row_index,
                )
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        subject=subject,
                        predicate=relation.relation,
                        object=obj,
                        properties={},
                        table_name=source_fragment.table_name,
                        row_index=source_fragment.row_index,
                        source_cluster=relation.source_cluster,
                        target_cluster=relation.target_cluster,
                    )
                )

        return triples

    def _build_graph(
        self,
        schema: GraphSchema,
        entities: dict[str, list[EntityInstance]],
        triples: list[Triple],
    ) -> tuple[Any, dict[str, list[str]]]:
        """Create the NetworkX MultiDiGraph from merged entities and relation triples."""
        graph = nx.MultiDiGraph(schema_name=schema.schema_name)
        nodes_by_class: dict[str, list[str]] = defaultdict(list)

        for class_id, instances in entities.items():
            for entity in instances:
                node_id = self._node_id(class_id, entity.id)
                graph.add_node(
                    node_id,
                    class_id=class_id,
                    entity_id=entity.id,
                    properties=entity.attributes,
                    source_fragments=entity.source_fragments,
                )
                nodes_by_class[class_id].append(node_id)

        for triple in triples:
            graph.add_edge(
                triple.subject,
                triple.object,
                key=triple.edge_id,
                edge_id=triple.edge_id,
                predicate=triple.predicate,
                properties=triple.properties,
                table_name=triple.table_name,
                row_index=triple.row_index,
                source_cluster=triple.source_cluster,
                target_cluster=triple.target_cluster,
            )

        return graph, dict(nodes_by_class)

    def _build_edge_index(self, triples: list[Triple]) -> dict[str, dict[str, Any]]:
        """Build a direct lookup for executor access to edge-bound properties."""
        return {
            triple.edge_id: {
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "properties": dict(triple.properties),
                "table_name": triple.table_name,
                "row_index": triple.row_index,
                "source_cluster": triple.source_cluster,
                "target_cluster": triple.target_cluster,
            }
            for triple in triples
        }

    @staticmethod
    def _get_pk_columns(cluster: ColumnCluster) -> list[str]:
        """Normalise primary_key_candidate to a list of column names."""
        pk = cluster.primary_key_candidate
        if isinstance(pk, list):
            return pk
        if pk:
            return [pk]
        return []

    def _extract_cluster_properties(
        self,
        row_values: dict[str, Any],
        cluster: ColumnCluster,
        entity: MergedEntity,
        table_name: str = "",
    ) -> dict[str, Any]:
        """Map raw row columns in this cluster into canonical entity properties."""
        cluster_columns = set(cluster.columns)
        properties: dict[str, Any] = {}

        for prop in entity.properties:
            value = self._pick_property_value(
                row_values=row_values,
                source_columns=prop.source_columns,
                qualified_source_columns=prop.qualified_source_columns,
                allowed_columns=cluster_columns,
                table_name=table_name,
            )
            if not self._is_missing(value):
                properties[prop.name] = value

        if not properties:
            for column in cluster.columns:
                value = row_values.get(column)
                if not self._is_missing(value):
                    properties[column] = value
        return properties

    def _extract_relation_properties(
        self,
        relation: Relation,
        source_fragment: ClusterInstance,
        target_fragment: ClusterInstance,
        table_name: str,
        row_index: int,
    ) -> dict[str, Any]:
        """Materialize row-level values for relation properties on one edge."""
        row_values = self._row_values_for_relation_fragments(
            source_fragment=source_fragment,
            target_fragment=target_fragment,
        )
        properties: dict[str, Any] = {}
        for prop in relation.properties:
            value = row_values.get(prop.source_column)
            if self._is_missing(value):
                continue
            properties[prop.name] = value
        if relation.properties and not properties:
            log.debug(
                "  [InstKG] No concrete relation properties found for %s (%s row=%d)",
                relation.relation,
                table_name,
                row_index,
            )
        return properties

    @staticmethod
    def _row_values_for_relation_fragments(
        source_fragment: ClusterInstance,
        target_fragment: ClusterInstance,
    ) -> dict[str, Any]:
        """Recover row-local values relevant to a relation from its endpoint fragments."""
        row_values: dict[str, Any] = {}
        row_values.update(source_fragment.row_values)
        row_values.update(target_fragment.row_values)
        return row_values

    @staticmethod
    def _pick_property_value(
        row_values: dict[str, Any],
        source_columns: list[str],
        allowed_columns: set[str],
        qualified_source_columns: list[str] | None = None,
        table_name: str = "",
    ) -> Any:
        """Pick the first non-empty row value among the allowed raw source columns.

        Prefer table-qualified provenance when available so two same-named raw
        columns from different tables (e.g. `SAVINGS.balance` vs `CHECKING.balance`)
        can never populate each other's properties.
        """
        if qualified_source_columns:
            for qualified in qualified_source_columns:
                column = qualified
                if "." in qualified:
                    source_table, _, raw_column = qualified.partition(".")
                    if table_name and source_table != table_name:
                        continue
                    column = raw_column
                if column not in allowed_columns:
                    continue
                value = row_values.get(column)
                if not InstanceKGBuilder._is_missing(value):
                    return value
            return None

        for column in source_columns:
            if column not in allowed_columns:
                continue
            value = row_values.get(column)
            if not InstanceKGBuilder._is_missing(value):
                return value
        return None

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Treat None and blank strings as missing."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    @staticmethod
    def _to_identifier(value: Any) -> str:
        """Normalize identifier values into stable string ids."""
        return str(value).strip()

    @staticmethod
    def _node_id(class_id: str, identifier_value: str) -> str:
        """Build the canonical graph node id."""
        return f"{class_id}:{identifier_value}"

    @staticmethod
    def _sortable_entity_key(entity: EntityInstance, primary_key: str | list[str]) -> tuple[int, Any]:
        """Sort numeric identifiers naturally when possible."""
        # For composite keys, use the first PK column for sorting.
        pk_attr: str = primary_key[0] if isinstance(primary_key, list) and primary_key else primary_key if isinstance(primary_key, str) else ""
        value = entity.attributes.get(pk_attr, entity.id) if pk_attr else entity.id
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))
