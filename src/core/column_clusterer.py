"""
Step 1: Column Clusterer

For each table in a QAItem, use an LLM to cluster its columns into
semantic groups.  Each group represents one conceptual entity (the
table's owner entity or a referenced entity), with one identifying
column per cluster.

Constraints:
  - Every column MUST appear in exactly one cluster.
  - Every cluster MUST have at least one identifying column (primary_key_candidate),
    which can be a single string or a list for composite keys.
  - Neighbouring columns tend to belong to the same cluster.
"""

from __future__ import annotations

from pydantic import Field

from src.core.table_understander import TableUnderstandingResult
from src.data.loader import QAItem, Table
from src.llm.client import LLMClient, get_llm
from src.llm.prompts import column_cluster as prompts
from src.utils.schema import ProjectModel
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ColumnCluster(ProjectModel):
    """One semantic cluster of columns within a table."""
    cluster_id: str                          # short snake_case id
    label: str                               # human-readable label
    description: str                         # what this cluster represents
    columns: list[str] = Field(default_factory=list)
    primary_key_candidate: str | list[str] = ""


class RelationProperty(ProjectModel):
    """One raw table column that belongs on a relation rather than an entity."""

    column_name: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)


class ClusteringResult(ProjectModel):
    """Result of clustering one table."""
    table_name: str
    table_description: str
    clusters: list[ColumnCluster] = Field(default_factory=list)
    relation_properties: list[RelationProperty] = Field(default_factory=list)

    @property
    def all_columns(self) -> set[str]:
        return (
            {c for cl in self.clusters for c in cl.columns}
            | {prop.column_name for prop in self.relation_properties}
        )

    def missing_columns(self, source_columns: list[str]) -> list[str]:
        """Return source columns that were not assigned to any cluster."""
        clustered_lower = {col.lower() for col in self.all_columns}
        return [col for col in source_columns if col.lower() not in clustered_lower]

    def duplicate_columns(self) -> list[str]:
        """Return columns that appear in more than one cluster."""
        counts: dict[str, tuple[str, int]] = {}
        for cluster in self.clusters:
            for col in cluster.columns:
                key = col.lower()
                if key in counts:
                    original, count = counts[key]
                    counts[key] = (original, count + 1)
                else:
                    counts[key] = (col, 1)
        for prop in self.relation_properties:
            key = prop.column_name.lower()
            if key in counts:
                original, count = counts[key]
                counts[key] = (original, count + 1)
            else:
                counts[key] = (prop.column_name, 1)
        return [original for original, count in counts.values() if count > 1]


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------

class ColumnClusterer:
    """Clusters columns within each table using an LLM."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def cluster_all(
        self,
        item: QAItem,
        understanding: TableUnderstandingResult,
    ) -> dict[str, ClusteringResult]:
        """Run column clustering on every table in the item.

        Returns a dict mapping table_name → ClusteringResult.
        """
        results: dict[str, ClusteringResult] = {}
        for tname, table in item.tables.items():
            log.info(
                "  [Cluster] Table '%s': %d columns, %d rows",
                tname, table.num_cols, table.num_rows,
            )
            results[tname] = self._cluster_one(tname, table, understanding)
        return results

    def _cluster_one(
        self,
        table_name: str,
        table: Table,
        understanding: TableUnderstandingResult,
    ) -> ClusteringResult:
        table_understanding = understanding.get_table(table_name)
        if table_understanding is None:
            raise ValueError(f"Missing table understanding for '{table_name}'")
        user_prompt = prompts.build_user_prompt(table_name, table, table_understanding)
        raw = self.llm.chat_json(prompts.SYSTEM_PROMPT, user_prompt)
        return self._parse(raw, table_name, table, table_understanding.description)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse(
        raw: dict,
        table_name: str,
        table: Table | None = None,
        table_description: str = "",
    ) -> ClusteringResult:
        result = ClusteringResult.model_validate({
            "table_name": table_name,
            "table_description": table_description,
            "clusters": raw.get("clusters", []),
            "relation_properties": raw.get("relation_properties", []),
        })
        seen_columns: set[str] = set()
        for cluster in result.clusters:
            pk = cluster.primary_key_candidate
            if not pk:
                log.warning(
                    "  [Cluster] '%s' in table '%s' has no identifying column!",
                    cluster.cluster_id or "?", table_name,
                )
            # ---- Inject PK columns missing from cluster.columns ----
            pk_cols = pk if isinstance(pk, list) else ([pk] if pk else [])
            for col in pk_cols:
                if col and col not in cluster.columns:
                    cluster.columns.append(col)
                    log.info(
                        "  [Cluster]   injected PK column '%s' into cluster '%s'",
                        col, cluster.cluster_id,
                    )
            seen_columns.update(cluster.columns)
        seen_columns.update(prop.column_name for prop in result.relation_properties)

        # ---- Validate relation_properties: deduplicate depends_on and
        #      reclassify single-entity dependencies as cluster attributes. ----
        valid_relation_props: list[RelationProperty] = []
        for prop in result.relation_properties:
            unique_deps = list(dict.fromkeys(prop.depends_on))
            if len(unique_deps) < 2:
                log.warning(
                    "  [Cluster] Table '%s': relation property '%s' has invalid "
                    "depends_on=%s (needs 2 distinct columns); reclassifying as "
                    "cluster attribute",
                    table_name, prop.column_name, prop.depends_on,
                )
                # Find the cluster containing the depends_on column
                target_cluster = None
                for dep_col in unique_deps:
                    for cluster in result.clusters:
                        if dep_col in cluster.columns:
                            target_cluster = cluster
                            break
                    if target_cluster:
                        break
                if target_cluster:
                    if prop.column_name not in target_cluster.columns:
                        target_cluster.columns.append(prop.column_name)
                        seen_columns.add(prop.column_name)
                else:
                    log.error(
                        "  [Cluster] Table '%s': cannot reclassify '%s' — "
                        "no cluster found for depends_on=%s",
                        table_name, prop.column_name, prop.depends_on,
                    )
            else:
                if unique_deps != prop.depends_on:
                    log.info(
                        "  [Cluster] Table '%s': deduplicated depends_on for "
                        "relation property '%s': %s -> %s",
                        table_name, prop.column_name,
                        prop.depends_on, unique_deps,
                    )
                    prop.depends_on = unique_deps
                valid_relation_props.append(prop)
        result.relation_properties = valid_relation_props

        # --- Validation ---
        if table is not None:
            original = {c.lower() for c in table.columns}
            clustered_lower = {c.lower() for c in seen_columns}

            missing = result.missing_columns(table.columns)
            extra = sorted(clustered_lower - original)
            duplicates = result.duplicate_columns()

            if missing:
                log.warning(
                    "  [Cluster] ⚠ Table '%s': %d column(s) MISSING from clusters: %s",
                    table_name, len(missing), missing,
                )
            if duplicates:
                log.warning(
                    "  [Cluster] ⚠ Table '%s': %d duplicate column(s) across clusters: %s",
                    table_name, len(duplicates), duplicates,
                )
            if extra:
                log.warning(
                    "  [Cluster] ⚠ Table '%s': %d unknown column(s) in clusters: %s",
                    table_name, len(extra), extra,
                )
            for prop in result.relation_properties:
                unknown_dependencies = [
                    col for col in prop.depends_on if col.lower() not in original
                ]
                if unknown_dependencies:
                    log.warning(
                        "  [Cluster] ⚠ Table '%s': relation property '%s' depends on unknown column(s): %s",
                        table_name,
                        prop.column_name,
                        unknown_dependencies,
                    )

            log.info(
                "  [Cluster] → %d clusters, %d relation properties, %d/%d columns covered"
                + (" (FULL)" if not missing and not duplicates else " (INCOMPLETE)"),
                len(result.clusters),
                len(result.relation_properties),
                len(clustered_lower & original),
                len(original),
            )
        else:
            log.info(
                "  [Cluster] → %d clusters, %d relation properties covering %d columns",
                len(result.clusters),
                len(result.relation_properties),
                len(seen_columns),
            )

        return result
