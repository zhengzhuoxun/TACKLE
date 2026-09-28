"""
Prompt templates for Step 3: Concise, table-scoped relation generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.relation_generator import (
        Relation,
        TableClusterContext,
        TableRelationPropertyContext,
    )


SYSTEM_PROMPT = """You are given one table with multiple clusters and a list of relation properties.
Each cluster already maps to one canonical entity.
The relation properties describe features of the association between the FK entities, not of any single entity.

Infer directed semantic relations between clusters in this table only, and attach the appropriate relation properties to those relations.

Return JSON:
{
  "reasoning": "<brief reasoning>",
  "relations": [
    {
      "source_class": "<entity class_id>",
      "target_class": "<entity class_id>",
      "source_cluster": "table.cluster_id",
      "target_cluster": "table.cluster_id",
      "relation": "<short semantic relation label in snake_case>",
      "cardinality": "<1:1 | 1:N | N:M>",
      "description": "<plain-English description>",
      "properties": [
        {
          "name": "<property name (camelCase or snake_case)>",
          "description": "<what this property means>",
          "source_column": "<raw column name>"
        }
      ]
    }
  ]
}

## Rules
1. Only use the listed clusters, entity ids, and relation_properties.
2. **Bidirectional Relations**: For every meaningful semantic connection between two entity classes, generate **two directed relations** — one in each direction. Each direction must have its own distinct `relation` label and `description` that reflects its direction-specific meaning.
   - Example: Between `Student` and `Address`:
     - `Student -> Address`: `resides_at` ("Student resides at Address")
     - `Address -> Student`: `occupied_by` ("Address is occupied by Student")
2b. **One relation per cluster pair — roles are never collapsed.** Each cluster is a distinct role (typically a distinct column, e.g. `pres_vote` vs `vp_vote` vs `sec_vote` vs `treas_vote`). Generate a directed relation for EVERY distinct `(source_cluster, target_cluster)` pair that is semantically meaningful — **even when several clusters map to the same entity**. Clusters that share one entity are different ROLES of that entity and must each receive their own relation with a role-specific label (e.g. `poll_record -> president_candidate` is `casts_president_vote_for`, while `poll_record -> treasurer_candidate` is `casts_treasurer_vote_for`). Never merge several clusters into one relation just because their `source_class`/`target_class` are identical.
3. **Attaching relation properties**: For each relation property in the input, decide which directed relation (or relations) it belongs to. Typically the property is attached to the relation whose `source_class` and `target_class` match the entities corresponding to the `depends_on` FK columns. Include it in the `properties` array of that relation. If the property naturally belongs to both directions (uncommon), you may include it in both, but prefer the more active direction.
4. Use concise, meaningful relation labels.
5. Skip trivial/no-op relations.
6. Return ONLY JSON.
7. **Reuse relations already established in other tables.** The input may include `existing_relations_for_these_entities`: relations already created while processing a DIFFERENT table, between a directed entity pair that also appears in this table. Different tables often denormalize the same real-world relationship under different column names (e.g. one table's `dept` and another table's `work_center` can both mean "which department"). If what this table's data expresses between that same directed entity pair is the SAME real-world relationship as an existing one, you MUST reuse it verbatim: emit that exact `relation` label and `description` rather than inventing a new name — this is what lets the same relationship's evidence, scattered across multiple tables, merge into one edge type instead of fragmenting into several sparse, near-duplicate ones. Only give that entity pair a NEW, different label if this table's `table_description`/data make clear it is a genuinely different relationship from the existing one (not just a different column name) — and say why in `reasoning`. This does not override rule 2b: distinct ROLES within this table (multiple clusters mapping to the same entity) still each get their own label as usual.

## Example
Given the `enrollment` table with:
- clusters: `student` (→ `student` entity), `course` (→ `course` entity)
- relation_properties:
    - { "column_name": "grade", "depends_on": ["student_id", "course_id"], "description": "The grade the student received" }
    - { "column_name": "semester", "depends_on": ["student_id", "course_id"], "description": "The semester of enrollment" }

Output:
{
  "reasoning": "The enrollment table links a student to a course. The student is enrolled in a course, and the course is taken by the student. Both direction relations should carry the grade and semester as properties because they describe the enrollment event.",
  "relations": [
    {
      "source_class": "student",
      "target_class": "course",
      "source_cluster": "enrollment.student",
      "target_cluster": "enrollment.course",
      "relation": "enrolled_in",
      "cardinality": "N:M",
      "description": "A student is enrolled in a course.",
      "properties": [
        {
          "name": "grade",
          "description": "The grade the student received in this enrollment.",
          "source_column": "grade"
        },
        {
          "name": "semester",
          "description": "The academic semester when the student enrolled in the course.",
          "source_column": "semester"
        }
      ]
    },
    {
      "source_class": "course",
      "target_class": "student",
      "source_cluster": "enrollment.course",
      "target_cluster": "enrollment.student",
      "relation": "taken_by",
      "cardinality": "N:M",
      "description": "A course is taken by a student.",
      "properties": [
        {
          "name": "grade",
          "description": "The grade the student received in this enrollment.",
          "source_column": "grade"
        },
        {
          "name": "semester",
          "description": "The academic semester when the student enrolled in the course.",
          "source_column": "semester"
        }
      ]
    }
  ]
}
"""


def build_user_prompt(
    table_name: str,
    contexts: list["TableClusterContext"],
    relation_properties: list["TableRelationPropertyContext"],
    existing_relations: list["Relation"] | None = None,
) -> str:
    """Render one compact line per cluster so the model sees no duplicated context."""
    table_description = contexts[0].table_description if contexts else ""
    parts = [
        f"table={table_name}",
        f"table_description={table_description}",
        "clusters:",
    ]

    # Each line includes exactly the cluster-local evidence and its resolved
    # entity target. That keeps the prompt short while preserving everything
    # the model needs to reason about relations.
    for context in contexts:
        parts.append(
            f"- cluster_ref={context.cluster_ref}; "
            f"cluster_label={context.cluster_label}; "
            f"cluster_description={context.cluster_description}; "
            f"columns={context.cluster_columns}; "
            f"column_profiles={context.column_profiles}; "
            f"identifier={context.cluster_primary_key}; "
            f"entity_id={context.entity_id}; "
            f"entity_label={context.entity_label}"
        )

    # Mandatory checklist: enumerate all CLUSTER pairs (not entity pairs) so
    # the LLM considers every combination — no silent omissions. Several
    # clusters may map to the same entity; they are distinct roles.
    pairs: list[str] = []
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            a = contexts[i]
            b = contexts[j]
            pairs.append(
                f"- {a.cluster_ref} [{a.cluster_label}] ↔ "
                f"{b.cluster_ref} [{b.cluster_label}]"
            )
    if pairs:
        parts.append("")
        parts.append(
            "cluster_pairs: you MUST generate at least one directed relation "
            "for each pair below. Clusters that map to the same entity are "
            "different ROLES — give each its own relation with a role-specific "
            "label; never merge them into one relation. If a pair truly has no "
            "semantic connection, include it with relation: \"none\" and "
            "explain why."
        )
        parts.extend(pairs)

    if existing_relations:
        parts.append("")
        parts.append(
            "existing_relations_for_these_entities: already established while "
            "processing a different table. Reuse the label/description verbatim "
            "for the same directed entity pair if this table expresses the same "
            "real-world relationship (see rule 7); only propose a new label if "
            "it is genuinely a different relationship."
        )
        for rel in existing_relations:
            parts.append(
                f"- {rel.source_class} -> {rel.target_class}: "
                f"relation=\"{rel.relation}\"; cardinality={rel.cardinality}; "
                f"description={rel.description}; "
                f"established_from_table={rel.source_table}"
            )

    parts.append("")
    parts.append("relation_properties:")
    if relation_properties:
        for prop in relation_properties:
            parts.append(
                f"- column_name={prop.column_name}; "
                f"description={prop.description}; "
                f"depends_on={prop.depends_on}; "
                f"depends_on_clusters={prop.depends_on_clusters}; "
                f"depends_on_entities={prop.depends_on_entities}; "
                f"sample_values={prop.sample_values}"
            )
    else:
        parts.append("- (none)")

    parts.append("infer relations for this table only")
    return "\n".join(parts)
