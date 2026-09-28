"""Advisory, schema-only validation of LLM-written Cypher.

Unlike the old validator this does NOT validate against a grounded plan — it
infers aliases and relationships from the query text itself and checks them
against the graph schema. It is ADVISORY: callers use it to record issues and
drive repair, never to reject a query that would otherwise execute.
"""

from __future__ import annotations

import re

from src.core.graph_schema_builder import GraphSchema
from src.cypher.identifiers import identifiers_missing_required_quotes

_REL_PATTERN = re.compile(
    r"\(\s*(?P<lnode>[^()]*?)\s*\)"
    r"\s*(?P<ldir><-)?\s*-?\s*\[(?P<rel>[^\[\]]*)\]\s*(?:->|-)?\s*"
    r"(?=\s*\((?P<rnode>[^()]*?)\s*\))"
)
_NODE_PATTERN = re.compile(
    r"\(\s*(?P<var>[A-Za-z_]\w*)?\s*(?::\s*(?P<label>`?[A-Za-z_]\w*`?))?\s*\)"
)
_REGEX_COMPARE_RE = re.compile(
    r"(?P<alias>[A-Za-z_]\w*)\s*\.\s*(?P<prop>`?[A-Za-z_]\w*`?)\s*=~"
)
_FIELD_RE = re.compile(r"(?P<alias>[A-Za-z_]\w*)\.(?P<prop>`?[A-Za-z_]\w*`?)")

_REGEX_UNSAFE_TYPES = {"numeric", "datetime", "date", "timestamp", "boolean", "bool"}


def _clean_ident(token: str | None) -> str | None:
    token = (token or "").strip()
    if len(token) >= 2 and token.startswith("`") and token.endswith("`"):
        token = token[1:-1]
    return token or None


def _parse_ref(text: str) -> tuple[str | None, str | None]:
    text = (text or "").strip()
    if ":" in text:
        var, label = text.split(":", 1)
        return _clean_ident(var), _clean_ident(label)
    return _clean_ident(text), None


def _schema_maps(schema: GraphSchema):
    node_id_by_name: dict[str, str] = {}
    node_attrs: dict[str, dict[str, str]] = {}
    for node in schema.nodes:
        node_id_by_name[node.node_id.lower()] = node.node_id
        node_id_by_name[node.label.lower()] = node.node_id
        attrs = {
            a.name.lower(): (a.semantic_type or "").strip().lower()
            for a in node.attributes
        }
        node_attrs[node.node_id.lower()] = attrs
        node_attrs[node.label.lower()] = attrs

    edge_set = {(e.source, e.label, e.target) for e in schema.edges}
    edge_props: dict[str, dict[str, str]] = {}
    for e in schema.edges:
        edge_props[e.label.lower()] = {
            a.name.lower(): (a.semantic_type or "").strip().lower()
            for a in e.properties
        }
    return node_id_by_name, node_attrs, edge_set, edge_props


def validate_cypher_schema(query: str, schema: GraphSchema | None) -> list[str]:
    """Return advisory issues for a raw Cypher query, checked against the schema."""
    query = (query or "").strip()
    if not query:
        return ["empty query"]
    if schema is None:
        return []

    node_id_by_name, node_attrs, edge_set, edge_props = _schema_maps(schema)
    issues: list[str] = []

    # alias -> node_id from labeled node patterns `(var:Label)`.
    alias_class: dict[str, str] = {}
    for m in _NODE_PATTERN.finditer(query):
        var = _clean_ident(m.group("var"))
        label = _clean_ident(m.group("label"))
        if not label:
            continue
        node_id = node_id_by_name.get(label.lower())
        if node_id is None:
            issues.append(f"unknown node label '{label}'")
            continue
        if var:
            alias_class[var] = node_id

    # Relationship patterns: register aliases and check edge existence/direction.
    alias_rel: dict[str, str] = {}
    for m in _REL_PATTERN.finditer(query):
        _, rel_label = _parse_ref(m.group("rel"))
        rel_var, _ = _parse_ref(m.group("rel"))
        if not rel_label:
            issues.append(f"relationship '{m.group(0)}' has no relation label")
            continue
        if rel_var:
            alias_rel[rel_var] = rel_label

        lcls = _node_class(m.group("lnode"), alias_class, node_id_by_name)
        rcls = _node_class(m.group("rnode"), alias_class, node_id_by_name)
        if not lcls or not rcls:
            continue  # cannot resolve endpoint classes; skip (advisory)
        if m.group("ldir") == "<-":
            ok = (rcls, rel_label, lcls) in edge_set
            arrow = f"{rcls}<-[{rel_label}]-{lcls}"
        else:
            ok = (lcls, rel_label, rcls) in edge_set
            arrow = f"{lcls}-[{rel_label}]->{rcls}"
        if not ok:
            issues.append(f"relationship '{arrow}' does not exist in the schema")

    # Property existence: only for aliases we resolved (avoids false positives
    # on literals/functions). `id` is the Kuzu node primary key.
    for m in _FIELD_RE.finditer(query):
        alias = m.group("alias")
        prop = _clean_ident(m.group("prop"))
        if prop is None or prop.lower() == "id":
            continue
        if alias in alias_rel:
            props = edge_props.get(alias_rel[alias].lower(), {})
        elif alias in alias_class:
            props = node_attrs.get(alias_class[alias].lower(), {})
        else:
            continue
        if prop.lower() not in props:
            issues.append(f"property '{alias}.{prop}' does not exist in the schema")

    # Regex `=~` must not be applied to non-string columns.
    for m in _REGEX_COMPARE_RE.finditer(query):
        alias = m.group("alias")
        prop = _clean_ident(m.group("prop"))
        stype = ""
        if alias in alias_rel:
            stype = edge_props.get(alias_rel[alias].lower(), {}).get(
                (prop or "").lower(), ""
            )
        elif alias in alias_class:
            stype = node_attrs.get(alias_class[alias].lower(), {}).get(
                (prop or "").lower(), ""
            )
        if stype in _REGEX_UNSAFE_TYPES:
            issues.append(
                f"regex '=~' cannot be applied to {stype} property "
                f"'{alias}.{prop}' (use a comparison operator)"
            )

    # Reserved identifiers must be backtick-quoted.
    identifiers: list[str] = []
    for node in schema.nodes:
        identifiers.extend([node.node_id, node.label])
        identifiers.extend(a.name for a in node.attributes)
    for edge in schema.edges:
        identifiers.append(edge.label)
        identifiers.extend(a.name for a in edge.properties)
    for name in identifiers_missing_required_quotes(query, identifiers):
        issues.append(f"reserved identifier '{name}' must be backtick-quoted")

    return issues


def _node_class(
    text: str,
    alias_class: dict[str, str],
    node_id_by_name: dict[str, str],
) -> str | None:
    """Resolve a node pattern's class from its label, alias, or bare label."""
    var, label = _parse_ref(text)
    if label:
        return node_id_by_name.get(label.lower())
    if var in alias_class:
        return alias_class[var]
    return node_id_by_name.get((var or "").lower())
