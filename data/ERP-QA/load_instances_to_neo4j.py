#!/usr/bin/env python3
"""
load_instances_to_neo4j.py

Load an instances.json (produced by generate_instance.py) into Neo4j as
labeled nodes and typed relationships, using the entity/relation names
declared in kgschema.yaml -- so the Cypher templates in
question_templates.yaml (e.g. ``(p:Plant)-[:has_department]->(d:Department)``)
resolve against real data instead of an empty or unrelated graph.

Every node and relationship this script creates carries an extra marker
label (default ``_SelfKG``), and the one destructive step here --
"delete everything from a previous load before reloading" -- is scoped to
that marker (``MATCH (n:_SelfKG) DETACH DELETE n``). This Neo4j instance may
be shared with unrelated data (other projects/graphs); the marker ensures
this script only ever touches nodes/relationships it created itself.

Node/relationship shape:
  * Entities with an ``id:`` block in kgschema.yaml (Machine, Product, Part,
    ProductionOrder) are keyed by their ``id`` property -- already globally
    unique.
  * Entities without an ``id:`` block (Plant, Department) are keyed by their
    own declared ``properties`` plus their full ``parent_<Ancestor>`` chain
    (as generate_instance.py already flattens onto every instance), so e.g.
    two different plants' "Testing" departments are distinct nodes.
  * Hierarchical ``parent:`` links (declared on the child entity in
    kgschema.yaml) become ``has_<child_entity_lower>`` edges from parent to
    child (matching question_templates.yaml's ``has_department``/
    ``has_machine`` convention).
  * Non-hierarchical relations (kgschema.yaml's top-level ``relations:``
    list) become edges named exactly as declared (``produce``, ``produceAt``,
    ``forProduct``, ``hasPart``, ``usePart``), with any relation properties
    (e.g. ``installation_date``) set on the edge.

After loading, if --templates is given (or found next to --schema), this
script does a static check: every ``(x:Label`` / ``[x:RelType]`` token
referenced across question_templates.yaml's Cypher must have been created by
this load, or a warning is printed -- catching schema drift (or a load that
silently produced an incomplete graph) before it turns into wrong ground
truth answers.

Usage:
    python load_instances_to_neo4j.py -i instances.json -s kgschema.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# ---------------------------------------------------------------------------
# Schema-driven key resolution
# ---------------------------------------------------------------------------

def own_identity_fields(entity_type: str, entities_schema: dict[str, Any]) -> list[str]:
    """The instance fields that, together with the parent chain, uniquely
    identify an instance of this entity type."""
    spec = entities_schema[entity_type]
    if spec.get("id"):
        return ["id"]
    # Same rule as generate_instance.py: the first property is the natural key.
    return [next(iter(spec.get("properties") or {}))]


def ancestor_chain(entity_type: str, entities_schema: dict[str, Any]) -> list[str]:
    """Ancestor type names from the immediate parent up to the root."""
    chain: list[str] = []
    cur = entities_schema[entity_type].get("parent")
    while cur:
        chain.append(cur)
        cur = entities_schema[cur].get("parent")
    return chain


def natural_key(entity_type: str, instance: dict[str, Any], entities_schema: dict[str, Any]) -> Any:
    """A hashable key uniquely identifying this instance, used to index its
    node for relationship resolution."""
    fields = own_identity_fields(entity_type, entities_schema)
    if fields == ["id"]:
        return instance["id"]
    parts = [(f, instance[f]) for f in fields]
    for anc in ancestor_chain(entity_type, entities_schema):
        parts.append((f"parent_{anc}", instance[f"parent_{anc}"]))
    return tuple(sorted(parts))


def parent_key_from_child(parent_type: str, child_instance: dict[str, Any], entities_schema: dict[str, Any]) -> Any:
    """Reconstruct parent_type's natural_key from a child instance's
    parent_<parent_type> (own identity value) and parent_<ancestor> fields
    (same field names, already flattened onto the child by generate_instance.py)."""
    fields = own_identity_fields(parent_type, entities_schema)
    if fields == ["id"]:
        return child_instance[f"parent_{parent_type}"]
    if len(fields) != 1:
        raise ValueError(
            f"Cannot resolve parent '{parent_type}': composite own-identity "
            f"{fields} is not supported for a parent_<type>-style reference."
        )
    parts = [(fields[0], child_instance[f"parent_{parent_type}"])]
    for anc in ancestor_chain(parent_type, entities_schema):
        parts.append((f"parent_{anc}", child_instance[f"parent_{anc}"]))
    return tuple(sorted(parts))


def flat_reference_field(entity_type: str, entities_schema: dict[str, Any]) -> str:
    """The field name whose bare value identifies this entity type when it
    appears as a relation's "from"/"to" endpoint (id, or -- for a rootless,
    single-property entity like Plant -- that property itself)."""
    fields = own_identity_fields(entity_type, entities_schema)
    if fields == ["id"]:
        return "id"
    if ancestor_chain(entity_type, entities_schema):
        raise ValueError(
            f"'{entity_type}' has no id and a parent chain; it cannot be "
            f"referenced by a bare relation endpoint value."
        )
    if len(fields) != 1:
        raise ValueError(
            f"'{entity_type}' has no id and multiple own properties {fields}; "
            f"it cannot be referenced by a bare relation endpoint value."
        )
    return fields[0]


def resolve_reference(
    entity_type: str,
    value: Any,
    node_index: dict[str, dict[Any, str]],
    entities_schema: dict[str, Any],
) -> str:
    field = flat_reference_field(entity_type, entities_schema)
    key = value if field == "id" else ((field, value),)
    eid = node_index.get(entity_type, {}).get(key)
    if eid is None:
        raise RuntimeError(f"Could not resolve {entity_type} reference {value!r} (key={key!r}).")
    return eid


# ---------------------------------------------------------------------------
# Neo4j write helpers (run inside a single explicit write transaction)
# ---------------------------------------------------------------------------

def _wipe_marker(tx, marker_label: str) -> None:
    tx.run(f"MATCH (n:`{marker_label}`) DETACH DELETE n")


def _create_node(tx, label: str, marker_label: str, props: dict[str, Any]) -> str:
    rec = tx.run(
        f"CREATE (n:`{label}`:`{marker_label}`) SET n = $props RETURN elementId(n) AS eid",
        props=props,
    ).single()
    return rec["eid"]


def _create_edge(tx, from_eid: str, rel_type: str, to_eid: str, props: dict[str, Any]) -> None:
    tx.run(
        "MATCH (a), (b) WHERE elementId(a) = $from_eid AND elementId(b) = $to_eid "
        f"CREATE (a)-[r:`{rel_type}`]->(b) SET r = $props",
        from_eid=from_eid, to_eid=to_eid, props=props,
    )


def _load(tx, schema: dict[str, Any], instances: dict[str, Any], marker_label: str) -> dict[str, Any]:
    entities_schema: dict[str, Any] = schema["entities"]
    relations_schema: list[dict[str, Any]] = schema.get("relations", []) or []

    _wipe_marker(tx, marker_label)

    node_counts: dict[str, int] = {}
    node_index: dict[str, dict[Any, str]] = {}
    # One representative raw instance per distinct key, used below to drive
    # edge creation exactly once per distinct node (not once per raw row).
    dedup_instances: dict[str, dict[Any, dict[str, Any]]] = {}
    duplicates: dict[str, int] = {}
    for entity_type, rows in instances.get("entities", {}).items():
        index: dict[Any, str] = {}
        reps: dict[Any, dict[str, Any]] = {}
        dup_count = 0
        for inst in rows:
            key = natural_key(entity_type, inst, entities_schema)
            if key in index:
                # Same identity (own properties + full parent chain) as an
                # instance already created -- e.g. two same-named Departments
                # under one Plant. Indistinguishable to any downstream
                # consumer (Cypher templates, flattened tables all match by
                # this same identity), so collapse onto the existing node
                # instead of silently dropping edges to/from either one.
                dup_count += 1
                continue
            eid = _create_node(tx, entity_type, marker_label, dict(inst))
            index[key] = eid
            reps[key] = inst
        node_index[entity_type] = index
        dedup_instances[entity_type] = reps
        node_counts[entity_type] = len(index)
        if dup_count:
            duplicates[entity_type] = dup_count

    edge_counts: dict[str, int] = {}

    # Hierarchical parent -> child links (has_<child_type>), one per distinct
    # child node.
    for entity_type, spec in entities_schema.items():
        parent_type = spec.get("parent")
        if not parent_type:
            continue
        rel_type = f"has_{entity_type.lower()}"
        count = 0
        for key, inst in dedup_instances.get(entity_type, {}).items():
            child_eid = node_index[entity_type][key]
            parent_key = parent_key_from_child(parent_type, inst, entities_schema)
            parent_eid = node_index.get(parent_type, {}).get(parent_key)
            if parent_eid is None:
                raise RuntimeError(
                    f"Could not resolve parent {parent_type} {parent_key!r} for {entity_type} instance {inst!r}."
                )
            _create_edge(tx, parent_eid, rel_type, child_eid, {})
            count += 1
        edge_counts[rel_type] = edge_counts.get(rel_type, 0) + count

    # Explicit non-hierarchical relations
    for rel in relations_schema:
        name = rel["name"]
        from_type, to_type = rel["from"], rel["to"]
        count = 0
        for inst in instances.get("relations", {}).get(name, []):
            from_eid = resolve_reference(from_type, inst["from"], node_index, entities_schema)
            to_eid = resolve_reference(to_type, inst["to"], node_index, entities_schema)
            props = {k: v for k, v in inst.items() if k not in ("from", "to")}
            _create_edge(tx, from_eid, name, to_eid, props)
            count += 1
        edge_counts[name] = edge_counts.get(name, 0) + count

    return {"nodes": node_counts, "edges": edge_counts, "duplicates": duplicates}


# ---------------------------------------------------------------------------
# Static consistency check against question_templates.yaml
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"\(\s*\w*\s*:\s*`?(\w+)`?")
_RELTYPE_RE = re.compile(r"\[\s*\w*\s*:\s*`?(\w+)`?")


def check_against_templates(templates_path: Path, node_counts: dict[str, int], edge_counts: dict[str, int]) -> None:
    with templates_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    labels_used: set[str] = set()
    reltypes_used: set[str] = set()
    for t in data.get("templates", []):
        cypher = t.get("cypher", "")
        labels_used.update(_LABEL_RE.findall(cypher))
        reltypes_used.update(_RELTYPE_RE.findall(cypher))

    missing_labels = sorted(l for l in labels_used if node_counts.get(l, 0) == 0)
    missing_rels = sorted(r for r in reltypes_used if edge_counts.get(r, 0) == 0)
    if missing_labels or missing_rels:
        print("[WARN] question_templates.yaml references labels/relationship types "
              "with zero instances in the loaded graph -- any template using them "
              "will silently produce empty/zero ground truth:")
        if missing_labels:
            print(f"         missing labels: {missing_labels}")
        if missing_rels:
            print(f"         missing relationship types: {missing_rels}")
    else:
        print("[OK] every label/relationship type referenced in "
              "question_templates.yaml has at least one instance in the loaded graph.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--instances", required=True, help="Path to instances.json")
    p.add_argument("-s", "--schema", required=True, help="Path to kgschema.yaml")
    p.add_argument("-t", "--templates", default=None,
                    help="Path to question_templates.yaml, for a post-load sanity check "
                         "(default: question_templates.yaml next to --schema, if present)")
    p.add_argument("--marker-label", default="_SelfKG",
                    help="Extra label stamped on every node this script creates; "
                         "the pre-load wipe only deletes nodes carrying it (default: _SelfKG)")
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="password")
    p.add_argument("--neo4j-database", default=None, help="Neo4j database name (default: server default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not installed. Run: pip install neo4j")

    schema_path = Path(args.schema)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(args.instances, "r", encoding="utf-8") as f:
        instances = json.load(f)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    try:
        with driver.session(database=args.neo4j_database) as session:
            result = session.execute_write(_load, schema, instances, args.marker_label)
    finally:
        driver.close()

    print(f"[DONE] Loaded into {args.neo4j_uri} (marker label: {args.marker_label})")
    print(f"       nodes: {result['nodes']}")
    print(f"       edges: {result['edges']}")
    if result["duplicates"]:
        print(f"[WARN] instances.json had entities with duplicate identity (same own "
              f"properties + parent chain) -- collapsed onto one node each, no "
              f"duplicate/orphaned edges created: {result['duplicates']}")
        print(f"       (this usually means generate_instance.py sampled a hierarchical "
              f"entity's values with replacement -- e.g. the same plant getting two "
              f"departments both named 'Machining' -- worth checking there too.)")

    templates_path = Path(args.templates) if args.templates else schema_path.parent / "question_templates.yaml"
    if templates_path.is_file():
        check_against_templates(templates_path, result["nodes"], result["edges"])


if __name__ == "__main__":
    main()
