"""
Step 3: Relation Generator

Relations are generated table by table. A table can only contribute relations
when it contains at least two clusters, because a single cluster has no local
counterpart to relate to.
"""

from __future__ import annotations

from pydantic import Field

from src.core.column_clusterer import ClusteringResult
from src.core.cluster_merger import MergedEntities
from src.core.table_understander import TableUnderstandingResult
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import relation_gen as prompts
from src.utils.logging import log
from src.utils.schema import ProjectModel


class Relation(ProjectModel):
    """One semantic relation between two classes with cluster provenance."""

    source_class: str
    target_class: str
    source_cluster: str
    target_cluster: str
    relation: str
    cardinality: str
    description: str
    properties: list["RelationPropertyAssignment"] = Field(default_factory=list)
    source_table: str = ""


class RelationPropertyAssignment(ProjectModel):
    """One property assigned to a semantic relation."""

    name: str
    description: str = ""
    source_column: str
    sample_values: list[object] = Field(default_factory=list)


class TableRelationRun(ProjectModel):
    """Debug metadata for one table-level relation generation pass."""

    table_name: str
    status: str
    reason: str = ""
    cluster_refs: list[str] = Field(default_factory=list)
    reasoning: str = ""
    relation_count: int = 0


class TableClusterContext(ProjectModel):
    """Minimal per-cluster context shared by prompting and validation."""

    table_description: str = ""
    cluster_ref: str
    cluster_label: str
    cluster_description: str = ""
    cluster_columns: list[str] = Field(default_factory=list)
    column_profiles: list[str] = Field(default_factory=list)
    cluster_primary_key: str | list[str] = ""
    entity_id: str
    entity_label: str


class TableRelationPropertyContext(ProjectModel):
    """Minimal per-property context shared by prompting and validation."""

    column_name: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    depends_on_clusters: list[str] = Field(default_factory=list)
    depends_on_entities: list[str] = Field(default_factory=list)
    sample_values: list[object] = Field(default_factory=list)


class RelationResult(ProjectModel):
    """Aggregate relation output across all eligible tables."""

    reasoning: str = ""
    relations: list[Relation] = Field(default_factory=list)
    tables_processed: list[str] = Field(default_factory=list)
    tables_skipped: list[TableRelationRun] = Field(default_factory=list)
    table_runs: list[TableRelationRun] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")


class _RawTableRelationResult(ProjectModel):
    """Validated shape of one table-scoped LLM response."""

    reasoning: str = ""
    relations: list[Relation] = Field(default_factory=list)


class RelationGenerator:
    """Identifies semantic relations between clusters/classes."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def generate(
        self,
        clusterings: dict[str, ClusteringResult],
        merged: MergedEntities,
        understanding: TableUnderstandingResult,
    ) -> RelationResult:
        """Generate relations one table at a time and aggregate the results."""
        log.info("  [Relate] Generating relations from %d tables", len(clusterings))

        result = RelationResult()
        reasoning_lines: list[str] = []
        seen_relation_keys: set[tuple[str, str, str, str]] = set()
        # Relations already established by earlier tables, keyed by directed
        # (source_class, target_class). Surfaced to later tables covering the
        # same entity pair so the model reuses one label per real-world
        # relation instead of each table inventing its own name for it.
        established_by_pair: dict[tuple[str, str], list[Relation]] = {}

        # Process each table independently so the model reasons only over
        # clusters that actually co-occur in the same table rows -- but pass
        # forward relations already established for the same entity pairs
        # (see established_by_pair) so identical relations don't fragment.
        for table_name, clustering in clusterings.items():
            if len(clustering.clusters) <= 1:
                skip = TableRelationRun(
                    table_name=table_name,
                    status="skipped",
                    reason="table has zero or one cluster",
                    cluster_refs=[
                        f"{table_name}.{cluster.cluster_id}"
                        for cluster in clustering.clusters
                    ],
                )
                result.tables_skipped.append(skip)
                result.table_runs.append(skip)
                log.info("  [Relate] Skipping '%s': %s", table_name, skip.reason)
                continue

            # Resolve one compact context object per cluster so the prompt
            # and validation logic can reuse the same authoritative view.
            contexts = self._build_table_cluster_context(
                table_name=table_name,
                clustering=clustering,
                merged=merged,
                understanding=understanding,
            )
            if len(contexts) != len(clustering.clusters):
                raise ValueError(
                    f"Table '{table_name}' has unmapped clusters during relation generation"
                )

            relation_properties = self._build_table_relation_property_context(
                table_name=table_name,
                clustering=clustering,
                contexts=contexts,
                understanding=understanding,
            )

            existing_relations = self._relevant_existing_relations(
                established_by_pair=established_by_pair,
                contexts=contexts,
            )

            table_result = self._generate_for_table(
                table_name=table_name,
                contexts=contexts,
                relation_properties=relation_properties,
                existing_relations=existing_relations,
            )
            validated_relations = self._validate_and_normalize_relations(
                table_name=table_name,
                relations=table_result.relations,
                contexts=contexts,
                relation_properties=relation_properties,
            )
            deduped_relations = self._dedupe_relations(
                relations=validated_relations,
                seen_keys=seen_relation_keys,
            )
            tagged_relations = [
                relation.model_copy(update={"source_table": table_name})
                for relation in deduped_relations
            ]
            self._register_established_relations(established_by_pair, tagged_relations)

            run = TableRelationRun(
                table_name=table_name,
                status="processed",
                reason="relations generated",
                cluster_refs=[context.cluster_ref for context in contexts],
                reasoning=table_result.reasoning,
                relation_count=len(tagged_relations),
            )
            result.tables_processed.append(table_name)
            result.table_runs.append(run)
            result.relations.extend(tagged_relations)
            reasoning_lines.append(f"{table_name}: {table_result.reasoning}")
            log.info(
                "  [Relate] Table '%s' -> %d relation(s)",
                table_name,
                len(tagged_relations),
            )

        result.reasoning = "\n".join(reasoning_lines)
        result.relations = self._ensure_unique_relation_labels(result.relations)
        log.info("  [Relate] → %d relations found", len(result.relations))
        return result

    def _build_table_cluster_context(
        self,
        table_name: str,
        clustering: ClusteringResult,
        merged: MergedEntities,
        understanding: TableUnderstandingResult,
    ) -> list[TableClusterContext]:
        """Resolve the global merge mapping into one compact per-cluster table view."""
        contexts: list[TableClusterContext] = []
        table_understanding = understanding.get_table(table_name)
        if table_understanding is None:
            raise ValueError(
                f"Table '{table_name}' has no understanding metadata during relation generation"
            )
        for cluster in clustering.clusters:
            cluster_ref = f"{table_name}.{cluster.cluster_id}"
            entity = merged.find_class_for_cluster(table_name, cluster.cluster_id)
            if entity is None:
                raise ValueError(
                    f"Cluster '{cluster_ref}' has no mapped entity from the merge step"
                )
            contexts.append(
                TableClusterContext(
                    table_description=table_understanding.description,
                    cluster_ref=cluster_ref,
                    cluster_label=cluster.label,
                    cluster_description=cluster.description,
                    cluster_columns=list(cluster.columns),
                    column_profiles=[
                        (
                            f"{column.column_name}: "
                            f"{column.semantic_type}; "
                            f"{column.description}; "
                            f"samples={column.sample_values}"
                        )
                        for column in table_understanding.columns
                        if column.column_name in cluster.columns
                    ],
                    cluster_primary_key=cluster.primary_key_candidate,
                    entity_id=entity.class_id,
                    entity_label=entity.label,
                )
            )
        return contexts

    def _build_table_relation_property_context(
        self,
        table_name: str,
        clustering: ClusteringResult,
        contexts: list[TableClusterContext],
        understanding: TableUnderstandingResult,
    ) -> list[TableRelationPropertyContext]:
        """Resolve relation-property FK dependencies against the current table clusters."""
        column_to_cluster: dict[str, str] = {}
        cluster_to_entity = {context.cluster_ref: context.entity_id for context in contexts}
        for context in contexts:
            for column_name in context.cluster_columns:
                column_to_cluster[column_name.lower()] = context.cluster_ref

        property_contexts: list[TableRelationPropertyContext] = []
        for prop in clustering.relation_properties:
            depends_on_clusters: list[str] = []
            depends_on_entities: list[str] = []
            for column_name in prop.depends_on:
                cluster_ref = column_to_cluster.get(column_name.lower())
                if cluster_ref is None:
                    log.warning(
                        "  [Relate] Table '%s': relation property '%s' depends on unmapped column '%s'",
                        table_name,
                        prop.column_name,
                        column_name,
                    )
                    continue
                depends_on_clusters.append(cluster_ref)
                depends_on_entities.append(cluster_to_entity[cluster_ref])

            property_contexts.append(
                TableRelationPropertyContext(
                    column_name=prop.column_name,
                    description=prop.description,
                    depends_on=list(prop.depends_on),
                    depends_on_clusters=_dedupe_preserve_order(depends_on_clusters),
                    depends_on_entities=_dedupe_preserve_order(depends_on_entities),
                    sample_values=self._lookup_relation_property_samples(
                        table_name=table_name,
                        column_name=prop.column_name,
                        understanding=understanding,
                    ),
                )
            )
        return property_contexts

    @staticmethod
    def _relevant_existing_relations(
        established_by_pair: dict[tuple[str, str], list["Relation"]],
        contexts: list[TableClusterContext],
    ) -> list["Relation"]:
        """Established relations whose endpoints are both entities in this table.

        These are surfaced to the model so it can reuse the same label/
        description for the same real-world relation instead of each table
        independently inventing its own name for it.
        """
        entity_ids = {context.entity_id for context in contexts}
        relevant: list[Relation] = []
        for (source_class, target_class), relations in established_by_pair.items():
            if source_class in entity_ids and target_class in entity_ids:
                relevant.extend(relations)
        return relevant

    @staticmethod
    def _register_established_relations(
        established_by_pair: dict[tuple[str, str], list["Relation"]],
        relations: list["Relation"],
    ) -> None:
        """Record this table's relations so later tables can see and reuse them."""
        for relation in relations:
            key = (relation.source_class, relation.target_class)
            bucket = established_by_pair.setdefault(key, [])
            if not any(
                existing.relation.strip().lower() == relation.relation.strip().lower()
                for existing in bucket
            ):
                bucket.append(relation)

    def _generate_for_table(
        self,
        table_name: str,
        contexts: list[TableClusterContext],
        relation_properties: list[TableRelationPropertyContext],
        existing_relations: list["Relation"] | None = None,
    ) -> _RawTableRelationResult:
        """Run one concise LLM call for one table using compact cluster context only."""
        user_prompt = prompts.build_user_prompt(
            table_name=table_name,
            contexts=contexts,
            relation_properties=relation_properties,
            existing_relations=existing_relations or [],
        )
        try:
            raw = self.llm.chat_json(prompts.SYSTEM_PROMPT, user_prompt)
            return _RawTableRelationResult.model_validate(raw)
        except Exception as exc:
            log.warning(
                "  [Relate] Table '%s' relation generation failed: %s",
                table_name,
                exc,
            )
            return _RawTableRelationResult(reasoning=f"fallback_empty: {exc}", relations=[])

    def _validate_and_normalize_relations(
        self,
        table_name: str,
        relations: list[Relation],
        contexts: list[TableClusterContext],
        relation_properties: list[TableRelationPropertyContext],
    ) -> list[Relation]:
        """Reject malformed relations and normalize class ids from the authoritative mapping."""
        allowed_clusters = {context.cluster_ref for context in contexts}
        cluster_to_entity = {
            context.cluster_ref: context.entity_id
            for context in contexts
        }
        property_by_column = {
            prop.column_name.lower(): prop
            for prop in relation_properties
        }

        validated: list[Relation] = []
        for relation in relations:
            if relation.source_cluster not in allowed_clusters:
                log.warning(
                    "  [Relate] Dropping relation with out-of-table source cluster '%s' in '%s'",
                    relation.source_cluster,
                    table_name,
                )
                continue
            if relation.target_cluster not in allowed_clusters:
                log.warning(
                    "  [Relate] Dropping relation with out-of-table target cluster '%s' in '%s'",
                    relation.target_cluster,
                    table_name,
                )
                continue
            if relation.source_cluster == relation.target_cluster:
                log.warning(
                    "  [Relate] Dropping self-relation on '%s' in '%s'",
                    relation.source_cluster,
                    table_name,
                )
                continue
            if (relation.relation or "").strip().lower() == "none":
                log.debug(
                    "  [Relate] Skipping explicit 'none' relation %s -> %s in '%s'",
                    relation.source_cluster, relation.target_cluster, table_name,
                )
                continue

            expected_source = cluster_to_entity[relation.source_cluster]
            expected_target = cluster_to_entity[relation.target_cluster]
            if relation.source_class != expected_source or relation.target_class != expected_target:
                log.debug(
                    "  [Relate] Normalizing class ids for %s -> %s",
                    relation.source_cluster,
                    relation.target_cluster,
                )

            # Always overwrite class ids from the authoritative mapping so
            # downstream steps see consistent entity references.
            normalized_properties = self._validate_relation_properties(
                table_name=table_name,
                relation=relation,
                property_by_column=property_by_column,
            )
            validated.append(
                relation.model_copy(
                    update={
                        "source_class": expected_source,
                        "target_class": expected_target,
                        "properties": normalized_properties,
                    }
                )
            )

        return validated

    def _validate_relation_properties(
        self,
        table_name: str,
        relation: Relation,
        property_by_column: dict[str, TableRelationPropertyContext],
    ) -> list[RelationPropertyAssignment]:
        """Keep only relation properties that are valid for the relation endpoints."""
        normalized: list[RelationPropertyAssignment] = []
        seen_columns: set[str] = set()
        relation_endpoints = {relation.source_cluster, relation.target_cluster}

        for prop in relation.properties:
            column_key = prop.source_column.lower()
            context = property_by_column.get(column_key)
            if context is None:
                log.warning(
                    "  [Relate] Dropping unknown relation property '%s' from relation '%s' in '%s'",
                    prop.source_column,
                    relation.relation,
                    table_name,
                )
                continue
            if context.depends_on_clusters and not set(context.depends_on_clusters).issubset(relation_endpoints):
                log.warning(
                    "  [Relate] Dropping relation property '%s' from '%s' in '%s' because depends_on=%s does not match endpoints %s",
                    prop.source_column,
                    relation.relation,
                    table_name,
                    context.depends_on_clusters,
                    sorted(relation_endpoints),
                )
                continue
            if column_key in seen_columns:
                continue
            seen_columns.add(column_key)
            normalized.append(
                RelationPropertyAssignment(
                    name=prop.name.strip() or context.column_name,
                    description=(
                        " ".join(prop.description.split()).strip()
                        or " ".join(context.description.split()).strip()
                    ),
                    source_column=context.column_name,
                    sample_values=list(context.sample_values),
                )
            )
        return normalized

    @staticmethod
    def _lookup_relation_property_samples(
        table_name: str,
        column_name: str,
        understanding: TableUnderstandingResult,
    ) -> list[object]:
        """Resolve Step-0 sample values for one relation-property column."""
        table = understanding.get_table(table_name)
        if table is None:
            return []
        profile = table.get_column(column_name)
        if profile is None:
            return []
        return list(profile.sample_values)

    def _dedupe_relations(
        self,
        relations: list[Relation],
        seen_keys: set[tuple[str, str, str, str]],
    ) -> list[Relation]:
        """Remove exact duplicates while preserving the first emitted relation."""
        deduped: list[Relation] = []
        for relation in relations:
            key = (
                relation.source_cluster,
                relation.target_cluster,
                relation.relation,
                relation.cardinality,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(relation)
        return deduped


    @staticmethod
    def _ensure_unique_relation_labels(
        relations: list[Relation],
    ) -> list[Relation]:
        """Deterministically enforce that every relation label maps to one relation.

        The first occurrence of each label keeps the original name and
        "claims" it for its (source_class, target_class) pair. A later
        relation reusing that exact label for the SAME directed class pair is
        left untouched -- that is intentional cross-table reuse of the same
        real-world relation (see established_by_pair in generate()), and
        keeping the label shared is what lets Step 5 merge their triples
        under one predicate. A later relation reusing the label for a
        DIFFERENT class pair is an accidental collision between two distinct
        relations and is renamed, as before.
        """
        seen_labels: set[str] = set()
        label_owner: dict[str, tuple[str, str]] = {}
        result: list[Relation] = []
        for relation in relations:
            label = relation.relation.strip()
            label_lower = label.lower()
            pair = (relation.source_class, relation.target_class)
            owner = label_owner.get(label_lower)

            if owner is None:
                seen_labels.add(label)
                label_owner[label_lower] = pair
                result.append(relation)
            elif owner == pair:
                result.append(relation)
            else:
                new_label = _unique_label_suffix(
                    label,
                    relation.source_class,
                    relation.target_class,
                    seen_labels,
                )
                log.info(
                    "  [Relate] Renamed duplicate label '%s' → '%s' (%s → %s)",
                    label,
                    new_label,
                    relation.source_class,
                    relation.target_class,
                )
                seen_labels.add(new_label)
                label_owner[new_label.lower()] = pair
                result.append(relation.model_copy(update={"relation": new_label}))
        return result

def _unique_label_suffix(
    label: str,
    source_class: str,
    target_class: str,
    seen_labels: set[str],
) -> str:
    """Try increasingly specific suffixes until the label is unique."""
    target = target_class.strip()
    source = source_class.strip()

    # Suffix 1: just target class
    candidate = f"{label}_{target}"
    if candidate.lower() not in {lbl.lower() for lbl in seen_labels}:
        return candidate

    # Suffix 2: source + target
    candidate = f"{label}_{source}_{target}"
    if candidate.lower() not in {lbl.lower() for lbl in seen_labels}:
        return candidate

    # Suffix 3: counter fallback
    counter = 2
    while True:
        candidate = f"{label}_{source}_{target}_{counter}"
        if candidate.lower() not in {lbl.lower() for lbl in seen_labels}:
            return candidate
        counter += 1


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
