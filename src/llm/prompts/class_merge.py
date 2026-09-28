"""
Prompt templates for Step 2: Incremental Class Merging.

Each cluster is processed one at a time against the current entity list.
The model decides whether to create a new entity or merge into an existing one,
and returns the resulting entity definition with canonical properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.column_clusterer import ColumnCluster
from src.core.table_understander import ColumnUnderstanding, ForeignKey

if TYPE_CHECKING:
    from src.core.cluster_merger import MergedEntity


SYSTEM_PROMPT = """You are an expert in entity resolution across database tables.

You are maintaining a growing list of canonical entities. You will be given:
1. The current entity list built so far.
2. One new cluster from a table.

Your task is to decide whether the incoming cluster:
- should CREATE a new entity, or
- should MERGE into exactly one existing entity.

The result must preserve and improve the canonical entity definition.

## Output Format
Return a JSON object:
{
  "reasoning": "<brief reasoning>",
  "action": "<create | merge>",
  "target_class_id": "<existing class_id if action=merge, else empty string>",
  "entity": {
    "class_id": "<short_snake_case>",
    "label": "<human-readable entity label>",
    "description": "<what this entity represents>",
    "primary_key": "<best identifying column>",
    "source_clusters": ["table.cluster_id", "..."],
    "all_columns": ["col_a", "col_b"],
    "properties": [
      {
        "name": "<canonical property name>",
        "description": "<what this property means>",
        "source_columns": ["raw_col_1", "raw_col_2"]
      }
    ]
  }
}

## Rules
1. If no existing entity is truly the same real-world entity, choose `create`.
2. If one existing entity is clearly the same entity, choose `merge`.
3. When merging, align incoming columns to existing properties where they mean the same thing.
4. Add new properties when the incoming cluster contributes new attributes.
5. `source_columns` should preserve the raw column names that support each canonical property. **Foreign key columns are not properties and must be omitted from properties and all_columns.**
6. `all_columns` should be the de-duplicated union of all raw columns in the entity (excluding foreign keys).
7. `source_clusters` should represent all clusters belonging to the entity.
8. Return ONLY JSON. When merging, inherit existing entity's properties and source_clusters; add new source clusters and any non-foreign-key columns.
9. **A matching identifier is necessary but NOT sufficient for merging.** Decide by what the cluster's non-identifier columns describe:
   - **Compare concepts, not just identifiers.** First name the real-world concept of the incoming cluster from its `cluster_label`/`cluster_description` (a person, an organization, a location, an event, a monetary amount, …). Merge only into an entity that represents the **same** concept. A person and an organization are never the same entity, even when their identifiers overlap.
   - **Sample values reveal the concept's granularity.** Inspect the sample values of both the incoming cluster and the candidate entity. Merge only if they describe the **same kind of thing at the same granularity**. Different categories or levels — e.g. country names (`France`, `USA`) vs city names (`Mexico City`, `Monterrey`), person names vs organisation names, product categories vs individual products — are **different concepts**: CREATE a new entity even when the labels look similar (`"Headquarters Location"` vs `"City"`).
   - If the cluster consists **only** of foreign-key/reference columns (it has no descriptive columns of its own):
       * If those foreign keys reference an **identifier** (`[primary_key]` or `[foreign_key]` in `cluster_foreign_keys`), merge the cluster into the referenced entity — the cluster IS that entity. A `[foreign_key]` target is still a key reference (e.g. a shared dimension code such as a country code), not a descriptive value.
       * If they reference a **descriptive** column (`[descriptive]`) or the target is `[unknown]`, the column is a shared VALUE, not a reference to an entity — do NOT merge into the referenced entity. Treat the column as an attribute of the cluster's own table's row entity (see rule 10's attribute exception).
   - If the cluster has its **own descriptive columns**, decide what real-world thing those columns describe:
       * If they describe the referenced entity itself (e.g. `person_name` describes the person), merge them into the referenced entity as properties.
       * If they describe a **different** thing that is merely associated with or held by the referenced entity (e.g. a balance of a savings/checking account, an order total, a loan amount), do **not** merge. **CREATE a new entity** for that thing, keyed by the foreign key. It will be linked to the referenced entity by a relation in the next step.
   - **A reference is not identity.** If an existing entity has a foreign-key column that points at the incoming cluster, that signals a *relationship* (the existing entity references the incoming cluster), not identity — do **not** merge on that basis alone. Keep them separate; the next step will create the relation. (Exception: the pure-foreign-key-stub case above still merges.)
   - **Sibling variants are distinct entities.** Two clusters that share the same identifier AND even the same raw column name are still different entities when their labels, descriptions, or table names denote different variants or roles of the same-looking concept (e.g. savings vs checking, personal vs business, income vs expense). Do not merge them just because their structure looks identical — **CREATE a separate entity for each variant**.
   - Examples:
       * Table A cluster `person(person_id)`, Table B cluster `person(person_id, person_name)` → MERGE into `person` (the name describes the person).
       * `EMPLOYEE` cluster `employee(EMP_NUM, EMP_FNAME, EMP_LNAME, …)` vs an existing `department` entity that has an `EMP_NUM` foreign key (the department head) → do NOT merge; CREATE `employee` (the `department → employee` relation is added in the next step).
       * `ACCOUNTS` cluster `customer(custid, name)`, `SAVINGS` cluster `savings_account_balance(custid, balance)` → do NOT merge; CREATE `savings_account` keyed by `custid` (the balance describes a savings account, not the customer).
       * `SAVINGS` cluster `savings_account_balance(custid, balance)` and `CHECKING` cluster `checking_account_balance(custid, balance)` → do NOT merge; CREATE `savings_account_balance` and `checking_account_balance` separately ("savings" vs "checking" are different roles).
       * `buildings` cluster `city(City)` with samples `["Mexico City", "Monterrey"]` vs `Companies` cluster `headquarters(Headquarters)` with samples `["France", "USA"]` → do NOT merge; CREATE `headquarters` (countries) as a separate entity.
10. **Clusters from the SAME table are distinct things that co-occur in one row, not the same entity — do NOT merge them.** The incoming cluster's `cluster_ref` and each existing entity's `source_clusters` carry the table name (e.g. `buildings.building`, `buildings.status`). When the incoming cluster shares a table with an existing entity, CREATE a new entity; the relation step will link them.
    - The ONLY same-table merge allowed is the **same concept in different roles**: e.g. `transfers.source_application` and `transfers.target_application` are both an `application` → MERGE into ONE `application` class, and let the relation step express the two roles (`source_of` vs `target_of`).
    - Same-table but different concepts are NEVER merged: `buildings.building` vs `buildings.status` → CREATE `status`; `buildings.building` vs `buildings.city` → CREATE `city`.
    - Attribute exception: a same-table cluster whose columns are ALL foreign keys to **descriptive** columns (`[descriptive]`/`[unknown]` in `cluster_foreign_keys`) is not a separate entity — it is an attribute of the table's own row entity (e.g. `wine_state` is an attribute of the wine, not a separate state entity). MERGE it into that row entity as a property instead of creating a new entity.

"""


def _render_entity(entity: "MergedEntity") -> str:
    props = []
    for prop in entity.properties:
        props.append(
            f"    - {prop.name}: source_columns={prop.source_columns}; "
            f"description={prop.description}; "
            f"sample_values={prop.sample_values}"
        )
    props_text = "\n".join(props) if props else "    - (no properties)"
    return (
        f"- class_id={entity.class_id}, label={entity.label}, "
        f"pk={entity.primary_key}, source_clusters={entity.source_clusters}, "
        f"all_columns={entity.all_columns}\n"
        f"  description={entity.description}\n"
        f"  properties:\n{props_text}"
    )


def build_user_prompt(
    entities: list["MergedEntity"],
    table_name: str,
    table_description: str,
    cluster: ColumnCluster,
    column_understanding: list[ColumnUnderstanding],
    foreign_keys: list[ForeignKey] | None = None,
    foreign_key_kinds: dict[str, str] | None = None,
) -> str:
    """Build the incremental merge prompt for one cluster."""
    cluster_ref = f"{table_name}.{cluster.cluster_id}"
    parts = ["## Current Canonical Entities"]
    if entities:
        for entity in entities:
            parts.append(_render_entity(entity))
    else:
        parts.append("(empty)")

    parts.append("")
    parts.append("## Incoming Cluster")
    parts.append(f"cluster_ref={cluster_ref}")
    parts.append(f"table_name={table_name}")
    parts.append(f"table_description={table_description}")
    parts.append(f"cluster_label={cluster.label}")
    parts.append(f"cluster_description={cluster.description}")
    parts.append(f"cluster_columns={cluster.columns}")
    parts.append(f"cluster_primary_key={cluster.primary_key_candidate}")
    parts.append("cluster_column_understanding:")
    for column in column_understanding:
        identifier = (
            f" identifier_kind={column.identifier_kind}"
            if column.identifier_kind
            else ""
        )
        parts.append(
            f"  - {column.column_name}: description={column.description}; "
            f"semantic_type={column.semantic_type}; "
            f"sample_values={column.sample_values}{identifier}"
        )
    parts.append("")
    parts.append(
        "cluster_foreign_keys (columns in this cluster that reference another "
        "table, the table/column they point to, and whether the target is an "
        "identifier [primary_key], another key reference [foreign_key], or a "
        "descriptive column [descriptive]):"
    )
    if foreign_keys:
        for fk in foreign_keys:
            target = (
                f"{fk.references_table}.{fk.references_column}"
                if fk.references_column
                else fk.references_table
            )
            kind = (foreign_key_kinds or {}).get(fk.column.lower(), "unknown")
            parts.append(f"  - {fk.column} -> {target} [{kind}]")
    else:
        parts.append("  - (none)")
    parts.append("")
    parts.append(
        "Decide whether to create a new entity or merge into one existing entity, "
        "then return the resulting canonical entity JSON."
    )
    return "\n".join(parts)
