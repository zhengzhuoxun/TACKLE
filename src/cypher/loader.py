"""InstanceKG → Kuzu loader.

Materialises the deterministic :class:`~src.core.instance_kg_builder.InstanceKG`
(NetworkX-backed) into an in-memory Kuzu database with typed node/relation
tables.

Type inference makes numeric equality work natively at query time: a value
``"4.0"`` stored in a numeric column is loaded as ``DOUBLE`` ``4.0``, so
``WHERE e.salary = 4`` matches it.
"""

from __future__ import annotations

from collections import defaultdict

import kuzu

from src.core.instance_kg_builder import InstanceKG
from src.cypher.comparison import to_iso_date, to_number
from src.cypher.identifiers import quote_identifier as _esc
from src.utils.logging import log


def infer_type(values: list) -> str:
    """Infer the single Kuzu column type for a list of raw values."""
    present = [v for v in values if v is not None]
    if not present:
        return "STRING"
    if all(isinstance(v, bool) for v in present):
        return "BOOL"
    nums = [to_number(v) for v in present]
    if all(n is not None for n in nums):
        if all(n.is_integer() for n in nums):
            return "INT64"
        return "DOUBLE"
    dates = [to_iso_date(v) for v in present]
    if all(d is not None for d in dates):
        return "DATE"
    return "STRING"


def _literal(value, ctype: str) -> str:
    if value is None:
        return "null"
    if ctype == "BOOL":
        if isinstance(value, bool):
            return "true" if value else "false"
        return "true" if str(value).strip().lower() in {"true", "1", "yes", "y"} else "false"
    if ctype == "INT64":
        n = to_number(value)
        return str(int(n)) if n is not None else "0"
    if ctype == "DOUBLE":
        n = to_number(value)
        return repr(float(n)) if n is not None else "0.0"
    if ctype == "DATE":
        d = to_iso_date(value)
        return f"DATE('{d.isoformat()}')" if d is not None else "null"
    return _str_literal(value)


def _str_literal(value) -> str:
    if value is None:
        return "null"
    s = str(value)
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _union_keys(attr_dicts: list[dict]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for attrs in attr_dicts:
        for key in attrs:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _normalize_attr_keys(attributes: dict) -> dict:
    """Return a copy with case-insensitive-deduplicated, lowercased keys.

    Kuzu identifiers are case-insensitive, so two keys that differ only in case
    (e.g. ``Station_ID`` vs ``station_id``) would collide as columns. Keep the
    first occurrence per lowercase key.
    """
    out: dict = {}
    seen: set = set()
    for key, value in attributes.items():
        lkey = key.lower()
        if lkey in seen:
            continue
        seen.add(lkey)
        out[lkey] = value
    return out


def _prop_set_clause(attributes: dict, type_map: dict) -> str:
    parts = []
    for key, value in attributes.items():
        parts.append(f"{_esc(key)}: {_literal(value, type_map.get(key, 'STRING'))}")
    return ", ".join(parts)


def load_to_kuzu(kg: InstanceKG) -> kuzu.Connection:
    """Load an instance KG into an in-memory Kuzu connection."""
    db = kuzu.Database(":memory:")
    conn = kuzu.Connection(db)

    # Instance KG node ids may be composite ("class:identifier") or bare
    # ("identifier"). Map both forms → class, and normalise to the canonical
    # composite id used as the stored node id.
    id_to_class: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for class_id, instances in kg.entities.items():
        for entity in instances:
            composite = f"{class_id}:{entity.id}"
            bare = str(entity.id)
            id_to_class[composite] = class_id
            id_to_class[bare] = class_id
            canonical[composite] = bare
            canonical[bare] = bare

    node_types: dict[str, dict[str, str]] = {}
    for class_id, instances in kg.entities.items():
        if not instances:
            continue
        # ``id`` is the reserved node primary key (set from ``entity.id``);
        # skip any source column also named ``id`` so it isn't duplicated.
        norm_attrs = [_normalize_attr_keys(e.attributes) for e in instances]
        prop_keys = [
            key for key in _union_keys(norm_attrs)
            if key != "id"
        ]
        type_map = {
            key: infer_type([attrs.get(key) for attrs in norm_attrs])
            for key in prop_keys
        }
        node_types[class_id] = type_map
        columns = ["id STRING"] + [
            f"{_esc(key)} {ctype}" for key, ctype in type_map.items()
        ]
        stmt = (
            f"CREATE NODE TABLE {_esc(class_id)}("
            + ", ".join(columns)
            + ", PRIMARY KEY(id))"
        )
        conn.execute(stmt)

    pred_map: dict[str, list] = defaultdict(list)
    for triple in kg.triples:
        pred_map[triple.predicate].append(triple)

    rel_types: dict[str, dict[str, str]] = {}
    for predicate, triples in pred_map.items():
        src_class = id_to_class.get(triples[0].subject)
        tgt_class = id_to_class.get(triples[0].object)
        if src_class is None or tgt_class is None:
            continue
        norm_props = [_normalize_attr_keys(t.properties) for t in triples]
        prop_keys = _union_keys(norm_props)
        type_map = {
            key: infer_type([props.get(key) for props in norm_props])
            for key in prop_keys
        }
        rel_types[predicate] = type_map
        columns = [f"{_esc(key)} {ctype}" for key, ctype in type_map.items()]
        body = ", ".join(columns)
        stmt = (
            f"CREATE REL TABLE {_esc(predicate)}("
            f"FROM {_esc(src_class)} TO {_esc(tgt_class)}"
            + (f", {body}" if body else "")
            + ")"
        )
        conn.execute(stmt)

    for class_id, instances in kg.entities.items():
        type_map = node_types.get(class_id, {})
        for entity in instances:
            # ``id`` is supplied separately from ``entity.id`` above.
            attrs = {
                key: value
                for key, value in _normalize_attr_keys(entity.attributes).items()
                if key != "id"
            }
            parts = [f"id: {_str_literal(entity.id)}"] + (
                [_prop_set_clause(attrs, type_map)]
                if attrs
                else []
            )
            stmt = f"CREATE (:{_esc(class_id)} {{{', '.join(parts)}}})"
            conn.execute(stmt)

    for triple in kg.triples:
        src_class = id_to_class.get(triple.subject)
        tgt_class = id_to_class.get(triple.object)
        if src_class is None or tgt_class is None:
            continue
        type_map = rel_types.get(triple.predicate, {})
        props = _prop_set_clause(_normalize_attr_keys(triple.properties), type_map)
        subject_id = canonical.get(triple.subject, triple.subject)
        object_id = canonical.get(triple.object, triple.object)
        match = (
            f"MATCH (a:{_esc(src_class)} {{id: {_str_literal(subject_id)}}}), "
            f"(b:{_esc(tgt_class)} {{id: {_str_literal(object_id)}}})"
        )
        create = f"CREATE (a)-[:{_esc(triple.predicate)}" + (f" {{{props}}}" if props else "") + "]->(b)"
        conn.execute(f"{match} {create}")

    log.info(
        "Loader | %d node tables, %d rel tables loaded",
        len(node_types), len(rel_types),
    )
    return conn
