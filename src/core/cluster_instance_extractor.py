"""
Cluster Instance Extractor

Runs immediately after column clustering (Step 1) to:
  1. Extract per-row cluster instances from raw table data.
  2. Validate that the same PK tuple always maps to the same property values.
     (This catches cluster definitions where the PK does not uniquely identify
     a consistent set of properties — the pipeline breaks early instead of
     silently producing corrupted instance KGs.)

Output: list[ClusterInstance] with class_id initialised to cluster_ref.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import Field

from src.core.column_clusterer import ClusteringResult, ColumnCluster
from src.data.loader import QAItem, Table
from src.utils.logging import log
from src.utils.schema import ProjectModel


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ClusterInstance(ProjectModel):
    """One row fragment produced from a single table cluster."""

    table_name: str
    row_index: int
    cluster_ref: str
    class_id: str
    identifier_column: str
    identifier_value: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_columns: list[str] = Field(default_factory=list)
    row_values: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClusterInstanceExtractor:
    """Extracts cluster instances from raw table rows and validates consistency."""

    def extract(
        self,
        item: QAItem,
        clusterings: dict[str, ClusteringResult],
    ) -> list[ClusterInstance]:
        """Extract cluster instances for all tables and validate PK consistency.

        Returns:
            list[ClusterInstance] — one fragment per (table, cluster, row).
            class_id is set to cluster_ref initially (resolved later after merging).
        """
        instances: list[ClusterInstance] = []

        for table_name in item.table_names:
            table = item.tables[table_name]
            clustering = clusterings.get(table_name)
            if clustering is None:
                raise ValueError(f"Missing clustering for table '{table_name}'")

            log.info(
                "  [InstExtract] Table '%s': %d clusters, %d rows",
                table_name, len(clustering.clusters), table.num_rows,
            )

            for cluster in clustering.clusters:
                cluster_ref = f"{table_name}.{cluster.cluster_id}"
                pk_cols = self._pk_columns(cluster)

                # Build per-row instances and group by PK for validation.
                cluster_instances: list[ClusterInstance] = []
                pk_groups: dict[tuple, list[ClusterInstance]] = defaultdict(list)

                for row_index, row in enumerate(table.rows):
                    row_values = self._row_dict(table, row)
                    pk_tuple = tuple(row_values.get(col) for col in pk_cols)
                    prop_values = self._property_values(cluster, row_values)

                    instance = ClusterInstance(
                        table_name=table_name,
                        row_index=row_index,
                        cluster_ref=cluster_ref,
                        class_id=cluster_ref,       # placeholder — resolved after merge
                        identifier_column=pk_cols[0] if pk_cols else "",
                        identifier_value=self._format_identifier(pk_tuple),
                        properties=prop_values,
                        source_columns=list(cluster.columns),
                        row_values=row_values,
                    )
                    instances.append(instance)
                    cluster_instances.append(instance)
                    pk_groups[pk_tuple].append(instance)

                # ---- PK expansion: when a single-column PK is insufficient,
                #      expand it by adding conflicting property columns. ----
                MAX_PK_COLUMNS = 3
                while True:
                    conflicts = self._find_pk_conflicts(pk_groups)
                    if not conflicts:
                        break
                    if len(pk_cols) >= MAX_PK_COLUMNS:
                        log.error(
                            "  [InstExtract] Table '%s' cluster '%s': PK expansion "
                            "limit (%d) reached; proceeding with imperfect PK %s",
                            table_name, cluster.cluster_id, MAX_PK_COLUMNS, pk_cols,
                        )
                        break

                    new_cols = sorted(
                        conflicts,
                        key=lambda col: table.columns.index(col)
                        if col in table.columns else 999,
                    )
                    pk_cols = list(pk_cols) + new_cols
                    cluster.primary_key_candidate = pk_cols

                    # Update all instances for this cluster
                    for inst in cluster_instances:
                        inst.identifier_column = pk_cols[0]
                        # Remove expanded PK columns from properties
                        inst.properties = {
                            k: v for k, v in inst.properties.items()
                            if k.lower() not in {c.lower() for c in new_cols}
                        }

                    # Re-group with expanded PK
                    pk_groups = defaultdict(list)
                    for inst in cluster_instances:
                        pk_tuple = tuple(
                            inst.row_values.get(col) for col in pk_cols
                        )
                        inst.identifier_value = self._format_identifier(pk_tuple)
                        pk_groups[pk_tuple].append(inst)

                    log.warning(
                        "  [InstExtract] Table '%s' cluster '%s': PK expanded to %s "
                        "(original PK not unique — %d conflicting column(s): %s)",
                        table_name, cluster.cluster_id, pk_cols,
                        len(new_cols), new_cols,
                    )

                # Final validation: should never fail after expansion.
                self._validate_property_consistency(
                    table_name, cluster, pk_cols, pk_groups,
                )

                log.info(
                    "  [InstExtract]   cluster '%s': %d unique PK values from %d rows",
                    cluster.cluster_id,
                    len(pk_groups),
                    table.num_rows,
                )

        log.info("  [InstExtract] → %d total cluster instances", len(instances))
        return instances

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pk_columns(cluster: ColumnCluster) -> list[str]:
        """Normalise primary_key_candidate to a list of column names."""
        pk = cluster.primary_key_candidate
        if isinstance(pk, list):
            return pk
        if pk:
            return [pk]
        return []

    @staticmethod
    def _row_dict(table: Table, row: list[Any]) -> dict[str, Any]:
        """Convert a raw row list into a {column_name: value} dict."""
        return {
            col: row[idx]
            for idx, col in enumerate(table.columns)
            if idx < len(row)
        }

    @staticmethod
    def _property_values(
        cluster: ColumnCluster,
        row_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract the (non-PK) property values for one cluster from a row."""
        pk_set = {col.lower() for col in ClusterInstanceExtractor._pk_columns(cluster)}
        return {
            col: row_values.get(col)
            for col in cluster.columns
            if col.lower() not in pk_set
        }

    @staticmethod
    def _format_identifier(pk_tuple: tuple) -> str:
        """Format a PK tuple into a stable string identifier."""
        if len(pk_tuple) == 1:
            val = pk_tuple[0]
            return str(val) if val is not None else ""
        return "|".join(str(v) if v is not None else "" for v in pk_tuple)

    @staticmethod
    def _find_pk_conflicts(
        pk_groups: dict[tuple, list[ClusterInstance]],
    ) -> set[str]:
        """Return property columns whose values differ within any PK group.

        A non-empty result means the current PK does not uniquely identify
        consistent property values and should be expanded.
        """
        conflicts: set[str] = set()
        for entries in pk_groups.values():
            if len(entries) <= 1:
                continue
            base = entries[0].properties
            for other in entries[1:]:
                for prop_col, base_val in base.items():
                    if other.properties.get(prop_col) != base_val:
                        conflicts.add(prop_col)
        return conflicts

    @staticmethod
    def _validate_property_consistency(
        table_name: str,
        cluster: ColumnCluster,
        pk_cols: list[str],
        pk_groups: dict[tuple, list[ClusterInstance]],
    ) -> None:
        """Raise ValueError if the same PK tuple maps to conflicting property values."""
        for pk_tuple, entries in pk_groups.items():
            if len(entries) <= 1:
                continue

            # Compare all entries against the first one.
            base = entries[0].properties
            for other in entries[1:]:
                for prop_col, base_val in base.items():
                    other_val = other.properties.get(prop_col)
                    if other_val != base_val:
                        raise ValueError(
                            f"Table '{table_name}' cluster '{cluster.cluster_id}': "
                            f"PK {pk_cols}={pk_tuple} has conflicting property '{prop_col}': "
                            f"Row {entries[0].row_index}: {base_val!r}  "
                            f"vs  Row {other.row_index}: {other_val!r}"
                        )

        # Also warn about rows with missing PK (None values).
        none_pk = pk_groups.get(tuple([None] * len(pk_cols)), [])
        if none_pk and len(pk_cols) > 0:
            log.warning(
                "  [InstExtract] ⚠ Table '%s' cluster '%s': %d row(s) have NULL PK %s",
                table_name, cluster.cluster_id, len(none_pk), pk_cols,
            )
