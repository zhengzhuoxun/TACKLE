"""Build a Kuzu-visible KG schema description for the direct-translation prompt.

The description mirrors exactly what :func:`src.cypher.loader.load_to_kuzu`
materialises — node tables named after each ``class_id`` (with an ``id`` primary
key), relationship tables named after each triple ``predicate``, and lowercased
property column names — so the LLM writes Cypher that runs first try. Property
samples are drawn deterministically from the actual :class:`InstanceKG`
entities/triples (first 3 distinct non-null values), which guarantees the
prompt's sample data is real rather than Step-0's possibly-empty guesses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.core.graph_schema_builder import GraphSchema
from src.core.instance_kg_builder import InstanceKG
from src.cypher.loader import infer_type


@dataclass
class ColumnDesc:
    """One Kuzu column: name, inferred type, and up to 3 sample values."""

    name: str
    ctype: str
    samples: list[Any] = field(default_factory=list)


@dataclass
class NodeTableDesc:
    """One Kuzu node table (``class_id``)."""

    table_name: str
    label: str
    description: str
    columns: list[ColumnDesc] = field(default_factory=list)


@dataclass
class RelTableDesc:
    """One Kuzu relationship table (``predicate``)."""

    table_name: str
    source: str
    target: str
    columns: list[ColumnDesc] = field(default_factory=list)


@dataclass
class KGSchemaDescription:
    """Structured Kuzu-visible schema description."""

    nodes: list[NodeTableDesc] = field(default_factory=list)
    relations: list[RelTableDesc] = field(default_factory=list)


def _normalize_keys(attributes: dict) -> dict:
    """Lowercase and case-insensitively dedupe attribute keys.

    Mirrors ``load_to_kuzu``'s column normalisation (Kuzu identifiers are
    case-insensitive, so ``Station_ID`` and ``station_id`` collide).
    """
    out: dict = {}
    seen: set = set()
    for key, value in attributes.items():
        lkey = str(key).lower()
        if lkey in seen:
            continue
        seen.add(lkey)
        out[lkey] = value
    return out


def _union_keys(attr_dicts: list[dict]) -> list[str]:
    """Union the keys of several attribute dicts, preserving first-seen order."""
    keys: list[str] = []
    seen: set = set()
    for attrs in attr_dicts:
        for key in attrs:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _distinct_samples(values: list, k: int = 3) -> list:
    """First ``k`` distinct non-null values, preserving order."""
    samples: list = []
    seen: set = set()
    for value in values:
        if value is None:
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        samples.append(value)
        if len(samples) >= k:
            break
    return samples


def _id_to_class(kg: InstanceKG) -> dict[str, str]:
    """Map both composite (``class:identifier``) and bare subject/object ids to
    their node class, matching the loader's identifier resolution."""
    mapping: dict[str, str] = {}
    for class_id, instances in kg.entities.items():
        for entity in instances:
            composite = f"{class_id}:{entity.id}"
            bare = str(entity.id)
            mapping[composite] = class_id
            mapping[bare] = class_id
    return mapping


def build_kg_description(
    kg: InstanceKG,
    schema: GraphSchema | None = None,
) -> KGSchemaDescription:
    """Derive the Kuzu-visible schema description from an instance KG."""
    node_meta = {n.node_id: n for n in schema.nodes} if schema else {}

    nodes: list[NodeTableDesc] = []
    for class_id, instances in kg.entities.items():
        if not instances:
            continue
        meta = node_meta.get(class_id)
        label = (meta.label if meta else None) or class_id.replace("_", " ").title()
        description = meta.description if meta else ""

        norm = [_normalize_keys(entity.attributes) for entity in instances]
        prop_keys = [key for key in _union_keys(norm) if key != "id"]

        id_samples = _distinct_samples([str(entity.id) for entity in instances])
        columns = [ColumnDesc("id", "STRING", id_samples)]
        for key in prop_keys:
            values = [attrs.get(key) for attrs in norm]
            columns.append(ColumnDesc(key, infer_type(values), _distinct_samples(values)))

        nodes.append(
            NodeTableDesc(
                table_name=class_id,
                label=label,
                description=description,
                columns=columns,
            )
        )

    id_to_class = _id_to_class(kg)
    pred_map: dict[str, list] = defaultdict(list)
    for triple in kg.triples:
        pred_map[triple.predicate].append(triple)

    relations: list[RelTableDesc] = []
    for predicate, triples in pred_map.items():
        source = id_to_class.get(triples[0].subject)
        target = id_to_class.get(triples[0].object)
        if source is None or target is None:
            continue
        norm_props = [_normalize_keys(t.properties) for t in triples]
        prop_keys = _union_keys(norm_props)
        columns: list[ColumnDesc] = []
        for key in prop_keys:
            values = [props.get(key) for props in norm_props]
            columns.append(
                ColumnDesc(key, infer_type(values), _distinct_samples(values))
            )
        relations.append(
            RelTableDesc(
                table_name=predicate,
                source=source,
                target=target,
                columns=columns,
            )
        )

    return KGSchemaDescription(nodes=nodes, relations=relations)


def _render_column(column: ColumnDesc) -> str:
    if column.samples:
        return f"{column.name} ({column.ctype})[samples={column.samples}]"
    return f"{column.name} ({column.ctype})"


def render_kg_description(desc: KGSchemaDescription) -> str:
    """Render the Kuzu-visible schema description as prompt text."""
    lines = ["### Node tables"]
    for node in desc.nodes:
        columns = ", ".join(_render_column(c) for c in node.columns) or "(none)"
        lines.append(f"- `{node.table_name}` (label: {node.label}): {node.description}")
        lines.append(f"  columns: {columns}")

    lines.append("")
    lines.append("### Relationship tables")
    if desc.relations:
        for rel in desc.relations:
            properties = ", ".join(_render_column(c) for c in rel.columns)
            lines.append(
                f"- `{rel.table_name}`: ({rel.source}) -[{rel.table_name}]-> ({rel.target})"
            )
            if properties:
                lines.append(f"  properties: {properties}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("### Connections")
    if desc.relations:
        for rel in desc.relations:
            lines.append(f"- ({rel.source}) --[{rel.table_name}]--> ({rel.target})")
    else:
        lines.append("- (none)")

    return "\n".join(lines)
