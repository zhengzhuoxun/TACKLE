#!/usr/bin/env python3
"""
Synthetic enterprise instance generator: entities first, then relations.

The output (instances.json) is the clean ground-truth graph of the benchmark:
every cardinality and consistency rule declared in kgschema.yaml holds, or
generation stops with an error. Messiness belongs to the table layer
(generate_tables.py / table_schema.yaml), not here.

Entities
  - Root entities: ``count`` instances. Without an ``id`` block, the first
    property is the natural key and is sampled WITHOUT replacement, so
    ``count`` must not exceed its number of ``values``.
  - Child entities: ``parent`` + ``count_per_parent: [min, max]``; natural keys
    are unique within each parent, or across ALL parents when the key property
    sets ``unique: true`` (e.g. a university has only one Physics department).
  - Property options (in addition to values / type+range / template):
      unique: true          on a non-key property: no value repeats within its
                            scope (per parent for child entities, e.g. room
                            numbers within a hotel; globally for root entities)
      aligned_with: <prop>  take the value at the same list position as <prop>'s
                            value, e.g. city aligned_with store_code, so VIE01 is
                            always in Vienna (both need `values` lists of equal
                            length)
  - ``--spec`` (specify.yaml) overrides the entity counts.

Relations (in the order they are declared)
  from_cardinality {min, max}: how many targets each source has.
  to_cardinality   {min, max}: how many sources each target has.
  (min defaults to 0, a missing max means unlimited.)

  1. Allowed pairs: all targets, or only those passing ``consistency``.
  2. Each source draws a quota in [from_min, from_max].
  3. Pass A: every target is linked until it reaches to_min.
  4. Pass B: every source is filled up to its quota, never exceeding to_max.
  5. Every rule is checked; on failure retry with new draws, then error out.

  Optional ``consistency`` requires source and target to share an anchor:
      consistency:
        source_anchor: parent_Plant        # read on the source instance
        target_anchor: hasPart.source      # read on the target instance
  An anchor path is a chain of steps, each either ``<relation>.source`` /
  ``<relation>.target`` (follow an already generated relation) or a field name
  (last step only). A pair is allowed when the two anchor sets overlap.

Reproducible with --seed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

MAX_ATTEMPTS = 50


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def generate_id(id_spec: dict, index: int) -> str:
    """Generate one ID from an ID-generation specification."""
    if "values" in id_spec:
        values = id_spec["values"]
        if not values:
            raise ValueError("ID 'values' cannot be empty.")
        if index >= len(values):
            raise ValueError(
                "More instances were requested than explicit ID values. "
                "Use prefix/digits or provide enough ID values."
            )
        return str(values[index])

    prefix = str(id_spec.get("prefix", ""))
    digits = int(id_spec.get("digits", 3))
    start = int(id_spec.get("start", 1))
    separator = str(id_spec.get("separator", ""))

    if digits < 1:
        raise ValueError("'digits' must be >= 1.")

    number = start + index
    return f"{prefix}{separator}{number:0{digits}d}"


def random_date(rng: random.Random, start: str, end: str) -> str:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if end_date < start_date:
        raise ValueError(f"Invalid date range: {start} > {end}")
    days = (end_date - start_date).days
    return (start_date + timedelta(days=rng.randint(0, days))).isoformat()


def generate_property(rng: random.Random, prop_name: str, spec: dict, context: dict) -> Any:
    """Generate a property value."""
    if spec is None:
        return None

    # Fixed vocabulary.
    values = spec.get("values", spec.get("choices"))
    if values is not None:
        if not values:
            raise ValueError(f"Property '{prop_name}' has an empty values list.")
        return rng.choice(values)

    # Template, e.g. "Machine {id}" or "{parent_plant_name} {id}".
    if "template" in spec:
        return str(spec["template"]).format(**context)

    prop_type = spec.get("type", "string")

    if prop_type == "integer":
        low, high = spec.get("range", [0, 100])
        return rng.randint(int(low), int(high))

    if prop_type == "float":
        low, high = spec.get("range", [0.0, 100.0])
        decimals = int(spec.get("decimals", 2))
        return round(rng.uniform(float(low), float(high)), decimals)

    if prop_type == "boolean":
        return rng.choice([True, False])

    if prop_type == "date":
        start, end = spec.get("range", ["2024-01-01", "2025-12-31"])
        return random_date(rng, start, end)

    if prop_type == "string":
        return f"{prop_name}_{context.get('id', '')}".rstrip("_")

    raise ValueError(f"Unsupported type '{prop_type}' for property '{prop_name}'.")


def get_entity_identifier(instance: dict) -> str:
    """
    Return the identifier of an entity instance.
    If it has an 'id' field, use that; otherwise, use the value of its first property.
    """
    if "id" in instance:
        return instance["id"]
    # No 'id' → assume the first key is the natural key (e.g., plant_name, department_name)
    first_key = next(iter(instance.keys()))
    return instance[first_key]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, updating base with override values."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _natural_key_values(entity_name: str, properties: dict) -> tuple[str, list]:
    """The natural-key property of an entity without an ``id`` block, and its
    vocabulary (the key must come from a fixed ``values`` list so it can be
    sampled without replacement)."""
    if not properties:
        raise ValueError(f"Entity '{entity_name}' has no ID and no properties to use as key.")
    key_prop = next(iter(properties.keys()))
    key_values = (properties[key_prop] or {}).get("values")
    if not key_values:
        raise ValueError(
            f"Key property '{key_prop}' of '{entity_name}' must have a 'values' list "
            "to ensure uniqueness when no ID is defined."
        )
    return key_prop, list(key_values)


def _validate_property_options(entity_name: str, properties: dict) -> None:
    """Check `unique` / `aligned_with` options before any sampling."""
    for prop_name, spec in properties.items():
        spec = spec or {}
        other = spec.get("aligned_with")
        if other is not None:
            other_spec = properties.get(other) or {}
            if other not in properties:
                raise ValueError(f"{entity_name}.{prop_name}: aligned_with '{other}' is not a property.")
            if other_spec.get("aligned_with"):
                raise ValueError(f"{entity_name}.{prop_name}: '{other}' is itself aligned; align both to the same property.")
            if not spec.get("values") or not other_spec.get("values") or len(spec["values"]) != len(other_spec["values"]):
                raise ValueError(
                    f"{entity_name}.{prop_name}: aligned_with needs `values` lists of equal length "
                    f"on both '{prop_name}' and '{other}'."
                )
        if spec.get("unique") and not spec.get("values"):
            raise ValueError(f"{entity_name}.{prop_name}: unique: true needs a `values` list.")


def _build_instance(
    rng: random.Random,
    id_spec: dict | None,
    index: int,
    key_prop: str | None,
    key_value: Any,
    properties: dict,
    context: dict,
    used: dict[str, set],
) -> dict:
    """Build one instance: id (or natural key) first, then the other properties.

    ``used`` holds the values already taken by `unique` properties in the
    current scope (one parent, or the whole entity for root entities).
    """
    if id_spec is not None:
        entity_id = generate_id(id_spec, index)
        context["id"] = entity_id
        instance = {"id": entity_id}
    else:
        instance = {key_prop: key_value}
    aligned = []
    for prop_name, prop_spec in properties.items():
        if id_spec is None and prop_name == key_prop:
            continue  # already assigned
        spec = prop_spec or {}
        if spec.get("aligned_with"):
            instance[prop_name] = None  # filled below, keeps column order
            aligned.append(prop_name)
        elif spec.get("unique"):
            taken = used.setdefault(prop_name, set())
            pool = [v for v in spec["values"] if v not in taken]
            if not pool:
                raise ValueError(
                    f"'{prop_name}' is unique: true but all {len(spec['values'])} values are "
                    "used in one scope; add values or lower the count."
                )
            instance[prop_name] = rng.choice(pool)
            taken.add(instance[prop_name])
        else:
            instance[prop_name] = generate_property(rng, prop_name, prop_spec, context)
    for prop_name in aligned:
        spec = properties[prop_name]
        other = spec["aligned_with"]
        position = properties[other]["values"].index(instance[other])
        instance[prop_name] = spec["values"][position]
    return instance


# ----------------------------------------------------------------------
# Entity generation
# ----------------------------------------------------------------------

def generate_entity(rng: random.Random, entity_name: str, spec: dict, result: dict) -> list[dict]:
    """
    Generate instances for one entity.

    Supports:
      - Independent entities (no 'parent')
      - Hierarchical entities with parent and count_per_parent
      - Optional ID generation (if 'id' block exists)
      - Unique natural keys for entities without ID (sampled without
        replacement: globally for root entities, per parent for children)
    """
    parent_entity = spec.get("parent")
    id_spec = spec.get("id")
    properties = spec.get("properties", {}) or {}
    _validate_property_options(entity_name, properties)
    key_prop, key_values = (None, [])
    if id_spec is None:
        key_prop, key_values = _natural_key_values(entity_name, properties)

    if parent_entity:
        parents = result.get(parent_entity)
        if not parents:
            raise ValueError(f"Parent entity '{parent_entity}' not found or empty.")

        count_per_parent = spec.get("count_per_parent")
        if count_per_parent is None:
            raise ValueError(f"'{entity_name}' has a parent but no count_per_parent.")

        instances = []
        local_index = 0
        # `unique: true` on the key: each value is used once across all parents.
        globally_unique = id_spec is None and bool((properties[key_prop] or {}).get("unique"))
        remaining = list(key_values)

        for parent in parents:
            used: dict[str, set] = {}  # unique properties: scope is one parent
            # Determine how many children for this parent
            if isinstance(count_per_parent, list):
                if len(count_per_parent) != 2:
                    raise ValueError(f"count_per_parent for '{entity_name}' must be [min, max].")
                min_c, max_c = map(int, count_per_parent)
                count = rng.randint(min_c, max_c)
            else:
                count = int(count_per_parent)

            # If no ID, sample unique key values for this parent
            if globally_unique:
                if count > len(remaining):
                    raise ValueError(
                        f"'{key_prop}' of '{entity_name}' is unique: true but has only "
                        f"{len(key_values)} values for all parents; add values or lower "
                        "count_per_parent."
                    )
                chosen_keys = rng.sample(remaining, count)
                remaining = [v for v in remaining if v not in chosen_keys]
            elif id_spec is None:
                if count > len(key_values):
                    raise ValueError(
                        f"Not enough distinct values for '{key_prop}' to create "
                        f"{count} instances under a single parent. Available: {len(key_values)}."
                    )
                chosen_keys = rng.sample(key_values, count)
            else:
                chosen_keys = [None] * count

            for i in range(count):
                # Parent context for templates
                context = {f"parent_{key}": value for key, value in parent.items()}
                instance = _build_instance(
                    rng, id_spec, local_index, key_prop, chosen_keys[i], properties, context, used
                )

                # Record parent linkage
                instance[f"parent_{parent_entity}"] = get_entity_identifier(parent)

                # Copy all parent_* fields from the immediate parent
                # This allows grandparent references to propagate (e.g., parent_Plant)
                for pkey, pvalue in parent.items():
                    if pkey.startswith("parent_"):
                        instance[pkey] = pvalue

                instances.append(instance)
                local_index += 1

        return instances

    # Independent entity (no parent)
    count = int(spec.get("count", 0))
    if count < 0:
        raise ValueError(f"Entity '{entity_name}' has negative count.")
    if id_spec is None:
        if count > len(key_values):
            raise ValueError(
                f"'{entity_name}' requests {count} instances but '{key_prop}' has only "
                f"{len(key_values)} distinct values; add values or lower the count."
            )
        chosen_keys = rng.sample(key_values, count)
    else:
        chosen_keys = [None] * count

    used: dict[str, set] = {}  # unique properties: scope is the whole entity
    return [
        _build_instance(rng, id_spec, index, key_prop, chosen_keys[index], properties, {}, used)
        for index in range(count)
    ]


def validate_unique_ids(result: dict):
    """Validate generated IDs within each entity type (only for entities that have 'id')."""
    for entity_name, instances in result.items():
        ids = [x["id"] for x in instances if "id" in x]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs found in entity '{entity_name}'.")


# ----------------------------------------------------------------------
# Relation generation
# ----------------------------------------------------------------------

def _cardinality(rel: dict, key: str) -> tuple[int, int | None]:
    card = rel.get(key) or {}
    return int(card.get("min") or 0), card.get("max")


def resolve_anchor(
    path: str,
    entity_type: str,
    identifier: str,
    entities: dict[str, list[dict]],
    relations_config: dict[str, dict],
    relation_edges: dict[str, list[dict]],
) -> set:
    """Follow an anchor path from one instance and return the set of values it reaches.

    Steps: ``<relation>.source`` / ``<relation>.target`` follow an already
    generated relation; a plain field name (last step only) reads that field.
    Without a trailing field, the reached instances' identifiers are returned.
    """
    by_id = {
        name: {get_entity_identifier(inst): inst for inst in insts}
        for name, insts in entities.items()
    }
    current_type, current_ids = entity_type, {identifier}
    steps = path.split(".")
    i = 0
    while i < len(steps):
        step = steps[i]
        if step in relations_config and i + 1 < len(steps) and steps[i + 1] in ("source", "target"):
            if step not in relation_edges:
                raise ValueError(
                    f"Anchor path '{path}' uses relation '{step}', which has not been "
                    "generated yet -- declare it earlier in the 'relations' list."
                )
            rel = relations_config[step]
            direction = steps[i + 1]
            if direction == "target":  # current instances are the relation's sources
                if rel["from"] != current_type:
                    raise ValueError(f"Anchor path '{path}': '{step}' does not start at {current_type}.")
                current_ids = {e["to"] for e in relation_edges[step] if e["from"] in current_ids}
                current_type = rel["to"]
            else:  # current instances are the relation's targets
                if rel["to"] != current_type:
                    raise ValueError(f"Anchor path '{path}': '{step}' does not end at {current_type}.")
                current_ids = {e["from"] for e in relation_edges[step] if e["to"] in current_ids}
                current_type = rel["from"]
            i += 2
            continue
        if i != len(steps) - 1:
            raise ValueError(f"Anchor path '{path}': field '{step}' must be the last step.")
        return {
            by_id[current_type][cid].get(step)
            for cid in current_ids
            if by_id[current_type][cid].get(step) is not None
        }
    return current_ids


def _allowed_pairs(
    rel: dict,
    from_ids: list[str],
    to_ids: list[str],
    entities: dict[str, list[dict]],
    relations_config: dict[str, dict],
    relation_edges: dict[str, list[dict]],
) -> dict[str, list[str]]:
    """For each source, the targets it may be linked to."""
    consistency = rel.get("consistency")
    if not consistency:
        return {s: list(to_ids) for s in from_ids}
    source_path = consistency.get("source_anchor")
    target_path = consistency.get("target_anchor")
    if not source_path or not target_path:
        raise ValueError(
            f"Relation '{rel['name']}': consistency needs both 'source_anchor' and 'target_anchor'."
        )
    target_anchors = {
        t: resolve_anchor(target_path, rel["to"], t, entities, relations_config, relation_edges)
        for t in to_ids
    }
    allowed = {}
    for s in from_ids:
        anchors = resolve_anchor(source_path, rel["from"], s, entities, relations_config, relation_edges)
        allowed[s] = [t for t in to_ids if anchors & target_anchors[t]]
    return allowed


def _sample_edges(
    rng: random.Random,
    from_ids: list[str],
    to_ids: list[str],
    allowed: dict[str, list[str]],
    from_min: int,
    from_max: int | None,
    to_min: int,
    to_max: int | None,
) -> list[tuple[str, str]] | None:
    """One sampling attempt. Returns the (source, target) pairs, or None when a
    minimum could not be met with these random draws."""
    # Hard per-source maximum (the declared max, capped by what is allowed) and
    # a random quota inside [from_min, that maximum].
    hard_max = {s: len(allowed[s]) if from_max is None else min(int(from_max), len(allowed[s])) for s in from_ids}
    quota = {s: rng.randint(min(from_min, hard_max[s]), hard_max[s]) for s in from_ids}
    cap = {t: float("inf") if to_max is None else int(to_max) for t in to_ids}
    sources_of = {t: [s for s in from_ids if t in allowed[s]] for t in to_ids}

    out_deg = Counter()
    in_deg = Counter()
    linked: set[tuple[str, str]] = set()

    def link(s: str, t: str) -> None:
        linked.add((s, t))
        out_deg[s] += 1
        in_deg[t] += 1

    # Pass A: bring every target up to to_min. Prefer sources still below
    # from_min, then sources below their quota, then any below their maximum.
    targets = list(to_ids)
    rng.shuffle(targets)
    for t in targets:
        while in_deg[t] < to_min:
            options = [s for s in sources_of[t] if (s, t) not in linked and out_deg[s] < hard_max[s]]
            if not options:
                return None
            for tier in (
                [s for s in options if out_deg[s] < from_min],
                [s for s in options if out_deg[s] < quota[s]],
                options,
            ):
                if tier:
                    link(rng.choice(tier), t)
                    break

    # Pass B: fill every source up to its quota, sources below from_min first.
    sources = list(from_ids)
    rng.shuffle(sources)
    sources.sort(key=lambda s: out_deg[s] >= from_min)
    for s in sources:
        while out_deg[s] < quota[s]:
            options = [t for t in allowed[s] if (s, t) not in linked and in_deg[t] < cap[t]]
            if not options:
                break
            link(s, rng.choice(options))

    if any(out_deg[s] < from_min for s in from_ids):
        return None
    from_pos = {s: i for i, s in enumerate(from_ids)}
    to_pos = {t: i for i, t in enumerate(to_ids)}
    return sorted(linked, key=lambda pair: (from_pos[pair[0]], to_pos[pair[1]]))


def generate_relations(
    rng: random.Random,
    entities: dict[str, list[dict]],
    relations_list: list[dict],
) -> dict[str, list[dict]]:
    """
    Generate non-hierarchical relations between entity instances, in declared
    order. Returns dict of relation_name -> list of edge dicts.
    """
    relations_config = {rel["name"]: rel for rel in relations_list}
    entity_ids = {name: [get_entity_identifier(inst) for inst in insts] for name, insts in entities.items()}
    relation_edges: dict[str, list[dict]] = {}

    for rel in relations_list:
        rel_name = rel["name"]
        from_ids = entity_ids[rel["from"]]
        to_ids = entity_ids[rel["to"]]
        from_min, from_max = _cardinality(rel, "from_cardinality")
        to_min, to_max = _cardinality(rel, "to_cardinality")

        allowed = _allowed_pairs(rel, from_ids, to_ids, entities, relations_config, relation_edges)
        starved = [s for s in from_ids if len(allowed[s]) < from_min]
        if starved:
            raise ValueError(
                f"Relation '{rel_name}': {len(starved)} {rel['from']} instance(s) have fewer than "
                f"from_cardinality.min={from_min} allowed {rel['to']} targets (e.g. {starved[0]})."
            )

        pairs = None
        for _ in range(MAX_ATTEMPTS):
            pairs = _sample_edges(rng, from_ids, to_ids, allowed, from_min, from_max, to_min, to_max)
            if pairs is not None:
                break
        if pairs is None:
            raise ValueError(
                f"Relation '{rel_name}': could not satisfy the cardinalities after {MAX_ATTEMPTS} "
                f"attempts ({len(from_ids)} {rel['from']} with from {from_min}..{from_max}, "
                f"{len(to_ids)} {rel['to']} with to {to_min}..{to_max}). Adjust counts or cardinalities."
            )

        edges = []
        for s, t in pairs:
            edge = {"from": s, "to": t}
            for pname, pspec in (rel.get("properties") or {}).items():
                edge[pname] = generate_property(rng, pname, pspec, {})
            edges.append(edge)
        relation_edges[rel_name] = edges

    return relation_edges


# ----------------------------------------------------------------------
# Validation report
# ----------------------------------------------------------------------

def validate_relations(
    entities: dict[str, list[dict]],
    relations_list: list[dict],
    relation_edges: dict[str, list[dict]],
) -> list[str]:
    """Check every cardinality and consistency rule; return one report line per
    relation and raise if any rule is violated."""
    relations_config = {rel["name"]: rel for rel in relations_list}
    entity_ids = {name: [get_entity_identifier(inst) for inst in insts] for name, insts in entities.items()}
    lines, problems = [], []

    for rel in relations_list:
        name = rel["name"]
        edges = relation_edges.get(name, [])
        from_ids, to_ids = entity_ids[rel["from"]], entity_ids[rel["to"]]
        out_deg = Counter(e["from"] for e in edges)
        in_deg = Counter(e["to"] for e in edges)
        outs = [out_deg[s] for s in from_ids]
        ins = [in_deg[t] for t in to_ids]
        from_min, from_max = _cardinality(rel, "from_cardinality")
        to_min, to_max = _cardinality(rel, "to_cardinality")

        if len({(e["from"], e["to"]) for e in edges}) != len(edges):
            problems.append(f"{name}: duplicate edges")
        if outs and (min(outs) < from_min or (from_max is not None and max(outs) > from_max)):
            problems.append(f"{name}: per-{rel['from']} degree {min(outs)}..{max(outs)} "
                            f"outside {from_min}..{from_max}")
        if ins and (min(ins) < to_min or (to_max is not None and max(ins) > to_max)):
            problems.append(f"{name}: per-{rel['to']} degree {min(ins)}..{max(ins)} "
                            f"outside {to_min}..{to_max}")

        violations = 0
        consistency = rel.get("consistency")
        if consistency:
            for e in edges:
                sa = resolve_anchor(consistency["source_anchor"], rel["from"], e["from"],
                                    entities, relations_config, relation_edges)
                ta = resolve_anchor(consistency["target_anchor"], rel["to"], e["to"],
                                    entities, relations_config, relation_edges)
                if not sa & ta:
                    violations += 1
            if violations:
                problems.append(f"{name}: {violations} consistency violation(s)")

        lines.append(
            f"  {name:<12} {len(edges):>4} edges | per {rel['from']}: "
            f"{min(outs, default=0)}..{max(outs, default=0)} (rule {from_min}..{from_max}) | "
            f"per {rel['to']}: {min(ins, default=0)}..{max(ins, default=0)} (rule {to_min}..{to_max})"
            + (f" | consistency violations: {violations}" if consistency else "")
        )

    if problems:
        raise ValueError("Generated relations violate the schema:\n  " + "\n  ".join(problems))
    return lines


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def generate_instances(config: dict, seed: int = 42) -> dict:
    """Generate entities and relations."""
    rng = random.Random(seed)
    entities_config = config.get("entities")
    if not entities_config:
        raise ValueError("YAML must contain an 'entities' section.")

    # Generate entities
    result = {}
    for entity_name, spec in entities_config.items():
        result[entity_name] = generate_entity(rng, entity_name, spec, result)

    validate_unique_ids(result)

    # Generate relations, then check every rule
    relations_list = config.get("relations", []) or []
    relations = generate_relations(rng, result, relations_list)
    report = validate_relations(result, relations_list, relations)

    return {"entities": result, "relations": relations}, report


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic enterprise instance data from YAML."
    )
    parser.add_argument("-i", "--input", required=True, help="Input YAML configuration.")
    parser.add_argument("-o", "--output", default="instances.json", help="Output JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--spec", help="Optional YAML file to override entity counts.")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply spec overrides if provided
    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {args.spec}")
        with spec_path.open("r", encoding="utf-8") as f:
            spec_data = yaml.safe_load(f)
        if spec_data and "entities" in spec_data:
            # Deep merge spec's entities into config's entities
            config["entities"] = deep_merge(config["entities"], spec_data["entities"])
            print("Applied entity count overrides from spec file.")
        else:
            print("Warning: spec file does not contain 'entities' section, ignoring.")

    output_data, report = generate_instances(config, seed=args.seed)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Generated data:")
    print(f"  Entities: {', '.join(f'{k}: {len(v)}' for k, v in output_data['entities'].items())}")
    print("  Relations (all cardinality and consistency rules checked):")
    print("\n".join(report))
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
