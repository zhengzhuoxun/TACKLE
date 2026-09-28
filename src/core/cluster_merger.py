"""
Step 2: Incremental Cluster Merger

Processes column clusters one by one and incrementally builds a list of
canonical entities. Each new cluster is either:
  - turned into a new entity, or
  - merged into one existing entity.
"""

from __future__ import annotations

from collections import OrderedDict

from pydantic import Field

from src.core import property_matcher
from src.core.column_clusterer import ClusteringResult, ColumnCluster
from src.core.table_understander import TableUnderstandingResult, ColumnUnderstanding, ForeignKey
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import class_merge as prompts
from src.utils.schema import ProjectModel
from src.utils.logging import log


class MergedEntity(ProjectModel):
    """One unified entity formed by merging equivalent clusters."""

    class_id: str
    label: str
    description: str
    primary_key: str | list[str] = ""
    source_clusters: list[str] = Field(default_factory=list)
    all_columns: list[str] = Field(default_factory=list)
    properties: list[EntityProperty] = Field(default_factory=list)

    def with_normalized_fields(self) -> "MergedEntity":
        """Return a copy with deduplicated columns and normalized properties."""
        props: list[EntityProperty] = []
        seen_prop_names: set[str] = set()
        for prop in self.properties:
            name = prop.name.strip() if prop.name else ""
            if not name:
                continue
            key = name.lower()
            if key in seen_prop_names:
                continue
            seen_prop_names.add(key)
            props.append(
                EntityProperty(
                    name=name,
                    description=prop.description.strip(),
                    source_columns=_dedupe_preserve_order(prop.source_columns),
                    qualified_source_columns=_dedupe_preserve_order(
                        prop.qualified_source_columns
                    ),
                    sample_values=list(prop.sample_values[:3]),
                )
            )

        all_columns = list(self.all_columns)
        for prop in props:
            all_columns.extend(prop.source_columns)

        return self.model_copy(
            update={
                "class_id": self.class_id.strip(),
                "label": self.label.strip(),
                "description": self.description.strip(),
                "primary_key": self.primary_key.strip() if isinstance(self.primary_key, str) else self.primary_key,
                "source_clusters": _dedupe_preserve_order(self.source_clusters),
                "all_columns": _dedupe_preserve_order(all_columns),
                "properties": props,
            }
        )


class MergeTrace(ProjectModel):
    """One incremental merge decision."""

    cluster_ref: str
    action: str
    target_class_id: str = ""
    reasoning: str = ""
    entity_snapshot: MergedEntity

class EntityProperty(ProjectModel):
    """A canonical property of a class, backed by one or more raw columns."""
    name: str
    description: str = ""
    source_columns: list[str] = Field(default_factory=list)
    qualified_source_columns: list[str] = Field(default_factory=list)
    sample_values: list[object] = Field(default_factory=list)
    
class MergedEntities(ProjectModel):
    """Result of incrementally merging clusters across all tables."""

    reasoning: str = ""
    classes: list[MergedEntity] = Field(default_factory=list)
    unmerged_clusters: list[str] = Field(default_factory=list)
    cluster_to_entity: dict[str, str] = Field(default_factory=dict)
    entity_to_clusters: dict[str, list[str]] = Field(default_factory=dict)
    merge_trace: list[MergeTrace] = Field(default_factory=list)

    def find_class_for_cluster(self, table: str, cluster_id: str) -> MergedEntity | None:
        """Given 'table.cluster_id', find the MergedEntity it was merged into."""
        ref = f"{table}.{cluster_id}"
        class_id = self.cluster_to_entity.get(ref)
        if class_id:
            return self.find_class_by_id(class_id)
        for cls in self.classes:
            if ref in cls.source_clusters:
                return cls
        return None

    def find_class_by_id(self, class_id: str) -> MergedEntity | None:
        for cls in self.classes:
            if cls.class_id == class_id:
                return cls
        return None

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")
    

class MergeDecision(ProjectModel):
    """LLM decision for one incoming cluster."""

    reasoning: str = ""
    action: str
    target_class_id: str = ""
    entity: MergedEntity


class ClusterCandidate(ProjectModel):
    """One cluster with table context for incremental processing."""

    table_name: str
    table_description: str
    cluster: ColumnCluster
    column_understanding: list[ColumnUnderstanding] = Field(default_factory=list)
    foreign_key_columns: set[str] = Field(default_factory=set)

    @property
    def cluster_ref(self) -> str:
        return f"{self.table_name}.{self.cluster.cluster_id}"


class ClassMerger:
    """Merges column clusters across tables into canonical entities."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def merge(
        self,
        clusterings: dict[str, ClusteringResult],
        understanding: TableUnderstandingResult,
    ) -> MergedEntities:
        """Merge clusters incrementally across tables."""
        candidates = self._flatten_clusters(clusterings, understanding)
        log.info(
            "  [Merge] Incrementally merging %d clusters across %d tables",
            len(candidates),
            len(clusterings),
        )

        result = MergedEntities()
        reasoning_lines: list[str] = []

        for candidate in candidates:
            decision = self._decide(candidate, result.classes, understanding, clusterings)
            applied_entity = self._apply_decision(candidate, decision, result)
            reasoning_lines.append(
                f"{candidate.cluster_ref}: {decision.action} -> {applied_entity.class_id}"
            )
            log.info(
                "  [Merge] %s -> %s (%s)",
                candidate.cluster_ref,
                decision.action,
                applied_entity.class_id,
            )

        result.reasoning = "\n".join(reasoning_lines)
        self._sync_entity_to_clusters(result)
        self._validate_mappings(result, candidates)

        log.info(
            "  [Merge] → %d unified classes from %d processed clusters",
            len(result.classes),
            len(candidates),
        )
        return result

    def _flatten_clusters(
        self,
        clusterings: dict[str, ClusteringResult],
        understanding: TableUnderstandingResult,
    ) -> list[ClusterCandidate]:
        candidates: list[ClusterCandidate] = []
        for table_name, clustering in clusterings.items():
            table_understanding = understanding.get_table(table_name)
            if table_understanding is None:
                raise ValueError(f"Missing table understanding for '{table_name}' during merge")
            for cluster in clustering.clusters:
                cluster_cols = {col.lower() for col in cluster.columns}
                foreign_key_columns = {
                    fk.column.lower()
                    for fk in table_understanding.foreign_keys
                    if fk.column.lower() in cluster_cols
                }
                candidates.append(
                    ClusterCandidate(
                        table_name=table_name,
                        table_description=clustering.table_description,
                        cluster=cluster,
                        column_understanding=[
                            column
                            for column in table_understanding.columns
                            if column.column_name in cluster.columns
                        ],
                        foreign_key_columns=foreign_key_columns,
                    )
                )
        return candidates

    def _decide(
        self,
        candidate: ClusterCandidate,
        entities: list[MergedEntity],
        understanding: TableUnderstandingResult,
        clusterings: dict[str, ClusteringResult],
    ) -> MergeDecision:
        table = understanding.get_table(candidate.table_name)
        cluster_columns = {col.lower() for col in candidate.cluster.columns}
        foreign_keys = [
            fk
            for fk in (table.foreign_keys if table else [])
            if fk.column.lower() in cluster_columns
        ]
        foreign_key_kinds = {
            fk.column.lower(): self._fk_target_kind(understanding, clusterings, fk)
            for fk in foreign_keys
        }
        user_prompt = prompts.build_user_prompt(
            entities=entities,
            table_name=candidate.table_name,
            table_description=candidate.table_description,
            cluster=candidate.cluster,
            column_understanding=candidate.column_understanding,
            foreign_keys=foreign_keys,
            foreign_key_kinds=foreign_key_kinds,
        )
        try:
            raw = self.llm.chat_json(prompts.SYSTEM_PROMPT, user_prompt)
            return MergeDecision.model_validate(raw)
        except Exception as exc:
            log.warning(
                "  [Merge] Decision parse failed for %s: %s; falling back to create",
                candidate.cluster_ref,
                exc,
            )
            return MergeDecision(
                reasoning=f"fallback_create: {exc}",
                action="create",
                target_class_id="",
                entity=self._entity_from_cluster(candidate),
            )

    @staticmethod
    def _fk_target_kind(
        understanding: TableUnderstandingResult,
        clusterings: dict[str, ClusteringResult],
        fk: ForeignKey,
    ) -> str:
        """Classify an FK target: primary_key / foreign_key (identifiers), descriptive, unknown.

        Step 0's ``identifier_kind`` only recognizes a column as an identifier
        when it uniquely identifies a RAW ROW of its own table. A denormalized
        dimension column (e.g. a `facility` column repeated on every row of a
        per-machine roster table) fails that test even though it uniquely
        identifies its own cluster's entity (the plant) -- which is exactly
        what Step 1 clustering's ``primary_key_candidate`` already captures
        correctly for that cluster. Falls back to that cluster-level signal
        before concluding the target is merely descriptive.
        """
        if not fk.references_column:
            return "unknown"
        ref_table = understanding.get_table(fk.references_table)
        if ref_table is None:
            return "unknown"
        ref_col = ref_table.get_column(fk.references_column)
        if ref_col is None:
            return "unknown"
        if ref_col.identifier_kind == "primary_key":
            return "primary_key"
        if ref_col.identifier_kind == "foreign_key":
            # A foreign_key target is still a key reference (e.g. a shared
            # dimension code), not a descriptive value.
            return "foreign_key"
        if ClassMerger._cluster_declares_identifier(
            clusterings, fk.references_table, fk.references_column
        ):
            return "primary_key"
        return "descriptive"

    @staticmethod
    def _cluster_declares_identifier(
        clusterings: dict[str, ClusteringResult],
        table_name: str,
        column_name: str,
    ) -> bool:
        """True if `column_name` is (part of) the primary_key_candidate of
        whichever cluster in `table_name` contains it."""
        clustering = clusterings.get(table_name)
        if clustering is None:
            return False
        key = column_name.lower()
        for cluster in clustering.clusters:
            if key not in {col.lower() for col in cluster.columns}:
                continue
            pk = cluster.primary_key_candidate
            pk_keys = (
                {p.lower() for p in pk if p}
                if isinstance(pk, list)
                else ({pk.lower()} if pk else set())
            )
            return key in pk_keys
        return False

    def _apply_decision(
        self,
        candidate: ClusterCandidate,
        decision: MergeDecision,
        result: MergedEntities,
    ) -> MergedEntity:
        cluster_ref = candidate.cluster_ref
        action = decision.action.strip().lower()
        fallback_entity = self._entity_from_cluster(candidate)

        if action not in {"create", "merge"}:
            log.warning(
                "  [Merge] Invalid action '%s' for %s; falling back to create",
                decision.action,
                cluster_ref,
            )
            action = "create"

        if action == "merge":
            target_id = decision.target_class_id.strip()
            existing = result.find_class_by_id(target_id)
            if existing is None:
                log.warning(
                    "  [Merge] Unknown merge target '%s' for %s; falling back to create",
                    target_id,
                    cluster_ref,
                )
                action = "create"
            else:
                updated = self._normalize_entity_for_candidate(
                    entity=decision.entity,
                    candidate=candidate,
                    existing=existing,
                )
                self._replace_class(result, existing.class_id, updated)
                result.cluster_to_entity[cluster_ref] = updated.class_id
                result.merge_trace.append(
                    MergeTrace(
                        cluster_ref=cluster_ref,
                        action="merge",
                        target_class_id=updated.class_id,
                        reasoning=decision.reasoning,
                        entity_snapshot=updated,
                    )
                )
                return updated

        created = self._normalize_entity_for_candidate(
            entity=decision.entity if action == "create" else fallback_entity,
            candidate=candidate,
            existing=None,
        )
        if not created.class_id:
            created = fallback_entity

        # Safety net: if the create produces a class_id that already exists
        # (e.g. two clusters both claim the same entity), do not append a
        # duplicate. Re-route through the merge path instead, blanking the
        # primary_key so the existing class's key is inherited. This keeps the
        # pipeline from aborting on a duplicate class_id while still merging
        # the candidate's columns into the existing class.
        collision = result.find_class_by_id(created.class_id)
        if collision is not None:
            log.warning(
                "  [Merge] Create for %s collides with existing class_id '%s'; merging instead",
                cluster_ref,
                created.class_id,
            )
            updated = self._normalize_entity_for_candidate(
                entity=decision.entity.model_copy(update={"primary_key": ""}),
                candidate=candidate,
                existing=collision,
            )
            self._replace_class(result, collision.class_id, updated)
            result.cluster_to_entity[cluster_ref] = updated.class_id
            result.merge_trace.append(
                MergeTrace(
                    cluster_ref=cluster_ref,
                    action="merge",
                    target_class_id=updated.class_id,
                    reasoning=decision.reasoning,
                    entity_snapshot=updated,
                )
            )
            return updated

        result.classes.append(created)
        result.cluster_to_entity[cluster_ref] = created.class_id
        result.merge_trace.append(
            MergeTrace(
                cluster_ref=cluster_ref,
                action="create",
                target_class_id=created.class_id,
                reasoning=decision.reasoning,
                entity_snapshot=created,
            )
        )
        return created

    def _normalize_entity_for_candidate(
        self,
        entity: MergedEntity,
        candidate: ClusterCandidate,
        existing: MergedEntity | None,
    ) -> MergedEntity:
        cluster = candidate.cluster
        cluster_ref = candidate.cluster_ref

        if existing is not None:
            class_id = existing.class_id
            label = entity.label or existing.label
            description = entity.description or existing.description
            primary_key = entity.primary_key or existing.primary_key or cluster.primary_key_candidate
            source_clusters = existing.source_clusters + [cluster_ref]
            properties = existing.properties + entity.properties
            all_columns = existing.all_columns + cluster.columns + entity.all_columns
        else:
            class_id = entity.class_id or _slugify(cluster.label or cluster.cluster_id)
            label = entity.label or cluster.label or class_id.replace("_", " ").title()
            description = entity.description or cluster.description
            primary_key = entity.primary_key or cluster.primary_key_candidate
            source_clusters = [cluster_ref]
            properties = entity.properties or self._properties_from_candidate(candidate)
            all_columns = cluster.columns + entity.all_columns

        # Ensure every raw cluster column is preserved in some property.
        # Track table-qualified provenance per property so same-named columns
        # from different tables never alias each other during instantiation.
        existing_by_name: dict[str, EntityProperty] = (
            {p.name.strip().lower(): p for p in existing.properties}
            if existing is not None
            else {}
        )
        properties = self._reconcile_property_names(
            properties, existing_by_name, candidate.foreign_key_columns
        )

        property_map: "OrderedDict[str, EntityProperty]" = OrderedDict()
        for prop in properties:
            key = prop.name.strip().lower()
            if not key:
                continue

            qualified = self._resolve_qualified_columns(
                prop=prop,
                candidate=candidate,
                existing_by_name=existing_by_name,
            )

            if key in property_map:
                merged_prop = property_map[key]
                property_map[key] = EntityProperty(
                    name=merged_prop.name,
                    description=merged_prop.description or prop.description,
                    source_columns=_dedupe_preserve_order(
                        merged_prop.source_columns + prop.source_columns
                    ),
                    qualified_source_columns=_dedupe_preserve_order(
                        merged_prop.qualified_source_columns + qualified
                    ),
                    sample_values=_pick_longer_sample_list(
                        merged_prop.sample_values,
                        prop.sample_values,
                    ),
                )
            else:
                property_map[key] = EntityProperty(
                    name=prop.name.strip(),
                    description=prop.description.strip(),
                    source_columns=_dedupe_preserve_order(prop.source_columns),
                    qualified_source_columns=_dedupe_preserve_order(qualified),
                    sample_values=list(prop.sample_values[:3]),
                )

        assigned = {
            col.lower()
            for prop in property_map.values()
            for col in prop.source_columns
        }
        for col in cluster.columns:
            if col.lower() not in assigned:
                property_map[col.lower()] = EntityProperty(
                    name=col,
                    description=self._column_description(candidate, col),
                    source_columns=[col],
                    qualified_source_columns=[self._qualify_column(candidate, col)],
                    sample_values=self._column_samples(candidate, col),
                )

        normalized = MergedEntity(
            class_id=class_id,
            label=label,
            description=description,
            primary_key=primary_key,
            source_clusters=source_clusters,
            all_columns=all_columns,
            properties=list(property_map.values()),
        ).with_normalized_fields()

        return normalized

    def _reconcile_property_names(
        self,
        properties: list[EntityProperty],
        existing_by_name: dict[str, EntityProperty],
        foreign_key_columns: set[str] | None = None,
    ) -> list[EntityProperty]:
        """Align incoming property names to existing properties semantically.

        When the LLM gave an incoming property a different canonical name than
        an equivalent existing property (e.g. ``stu_nr`` vs ``student_id``),
        rename it to the existing name so the downstream name-keyed merge
        collapses them into one property.

        FK/role columns are exempt from fuzzy alignment: two distinct FK
        columns with similar names (e.g. ``class_sen_vote`` vs
        ``class_pres_vote``) are different roles and must stay separate, so
        they keep their own raw column name.
        """
        if not existing_by_name:
            return properties

        reconciled: list[EntityProperty] = []
        for prop in properties:
            name = (prop.name or "").strip()
            if not name and prop.source_columns:
                name = prop.source_columns[0]
            if not name:
                reconciled.append(prop)
                continue

            key = name.lower()
            if key not in existing_by_name:
                is_fk_prop = (
                    bool(prop.source_columns)
                    and bool(foreign_key_columns)
                    and all(
                        col.lower() in foreign_key_columns
                        for col in prop.source_columns
                    )
                )
                if not is_fk_prop:
                    matched = property_matcher.match_property(
                        prop,
                        list(existing_by_name.values()),
                    )
                    if matched is not None:
                        log.info(
                            "  [Merge] Aligned property '%s' -> existing '%s' (%.3f)",
                            name,
                            matched.name,
                            property_matcher.property_similarity(prop, matched),
                        )
                        name = matched.name.strip()

            reconciled.append(
                EntityProperty(
                    name=name,
                    description=prop.description.strip(),
                    source_columns=_dedupe_preserve_order(prop.source_columns),
                    qualified_source_columns=_dedupe_preserve_order(
                        prop.qualified_source_columns
                    ),
                    sample_values=list(prop.sample_values[:3]),
                )
            )
        return reconciled

    def _replace_class(
        self,
        result: MergedEntities,
        class_id: str,
        updated: MergedEntity,
    ) -> None:
        for idx, existing in enumerate(result.classes):
            if existing.class_id == class_id:
                result.classes[idx] = updated
                return
        result.classes.append(updated)

    def _entity_from_cluster(self, candidate: ClusterCandidate) -> MergedEntity:
        cluster = candidate.cluster
        return MergedEntity(
            class_id=_slugify(cluster.label or cluster.cluster_id),
            label=cluster.label or cluster.cluster_id,
            description=cluster.description,
            primary_key=cluster.primary_key_candidate,
            source_clusters=[candidate.cluster_ref],
            all_columns=list(cluster.columns),
            properties=self._properties_from_candidate(candidate),
        ).with_normalized_fields()

    def _properties_from_candidate(self, candidate: ClusterCandidate) -> list[EntityProperty]:
        cluster = candidate.cluster
        return [
            EntityProperty(
                name=column,
                description=self._column_description(candidate, column),
                source_columns=[column],
                qualified_source_columns=[self._qualify_column(candidate, column)],
                sample_values=self._column_samples(candidate, column),
            )
            for column in cluster.columns
        ]

    @staticmethod
    def _qualify_column(candidate: ClusterCandidate, column_name: str) -> str:
        """Qualify a raw column name with its source table to disambiguate it."""
        return f"{candidate.table_name}.{column_name}"

    @classmethod
    def _resolve_qualified_columns(
        cls,
        prop: EntityProperty,
        candidate: ClusterCandidate,
        existing_by_name: dict[str, EntityProperty],
    ) -> list[str]:
        """Resolve table-qualified provenance for one property.

        Already-qualified properties keep their provenance. LLM-provided
        properties fall into two cases:
          - inherited property (name matches an existing property) → union the
            existing property's qualified columns with the incoming columns;
          - new property → qualify each raw source column with the current
            candidate table.
        """
        if prop.qualified_source_columns:
            return list(prop.qualified_source_columns)

        key = prop.name.strip().lower()
        base = existing_by_name.get(key) if key else None
        incoming = [
            cls._qualify_column(candidate, col)
            for col in prop.source_columns
        ]
        if base is not None:
            return _dedupe_preserve_order(
                list(base.qualified_source_columns) + incoming
            )
        return incoming

    @staticmethod
    def _column_description(candidate: ClusterCandidate, column_name: str) -> str:
        """Resolve a Step-0 column description for one raw column."""
        key = column_name.lower()
        for column in candidate.column_understanding:
            if column.column_name.lower() == key:
                return column.description
        return ""

    @staticmethod
    def _column_samples(candidate: ClusterCandidate, column_name: str) -> list[object]:
        """Resolve Step-0 sample values for one raw column."""
        key = column_name.lower()
        for column in candidate.column_understanding:
            if column.column_name.lower() == key:
                return list(column.sample_values)
        return []

    @staticmethod
    def _sync_entity_to_clusters(result: MergedEntities) -> None:
        entity_to_clusters: dict[str, list[str]] = {}
        for cls in result.classes:
            entity_to_clusters[cls.class_id] = list(cls.source_clusters)
        result.entity_to_clusters = entity_to_clusters

    @staticmethod
    def _validate_mappings(
        result: MergedEntities,
        candidates: list[ClusterCandidate],
    ) -> None:
        expected = {candidate.cluster_ref for candidate in candidates}
        actual = set(result.cluster_to_entity)
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"Missing cluster-to-entity mappings for: {missing}")

        for cluster_ref, class_id in result.cluster_to_entity.items():
            cls = result.find_class_by_id(class_id)
            if cls is None:
                raise ValueError(
                    f"Cluster '{cluster_ref}' points to missing class '{class_id}'"
                )
            if cluster_ref not in cls.source_clusters:
                raise ValueError(
                    f"Cluster '{cluster_ref}' missing from source_clusters of '{class_id}'"
                )


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


def _pick_longer_sample_list(left: list[object], right: list[object]) -> list[object]:
    left_samples = list(left[:3])
    right_samples = list(right[:3])
    if len(right_samples) > len(left_samples):
        return right_samples
    return left_samples


def _slugify(text: str) -> str:
    chars: list[str] = []
    last_was_sep = False
    for ch in text.strip().lower():
        if ch.isalnum():
            chars.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    value = "".join(chars).strip("_")
    return value or "entity"
