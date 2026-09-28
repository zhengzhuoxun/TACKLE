#!/usr/bin/env python3
"""
Convert instance KG into denormalized tables (CSV) based on a schema.
Additionally, output a JSON file describing primary keys and foreign keys.
"""

import argparse
import csv
import json
from pathlib import Path

import yaml


def load_instances(instances_path: Path) -> dict:
    with instances_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_entity_by_id(entities, entity_type, entity_id):
    """Find an entity instance by its identifier."""
    for inst in entities.get(entity_type, []):
        if get_entity_identifier(inst) == entity_id:
            return inst
    return None


def get_entity_identifier(instance: dict) -> str:
    if "id" in instance:
        return instance["id"]
    first_key = next(iter(instance.keys()))
    return instance[first_key]


def _edge_for(rel_name: str, context: dict, direction: str | None):
    """The edge of relation ``rel_name`` for the current row.

    The row's own edge is used only when it belongs to ``rel_name``;
    otherwise the edge is looked up from the current entity: as its source
    when ``direction`` is 'target', as its target when 'source', and either
    way round when ``direction`` is None (a relation-property column).
    """
    if context.get('relation') is not None and context.get('relation_name') == rel_name:
        return context['relation']
    if context.get('entity') is None:
        return None
    cur_id = get_entity_identifier(context['entity'])
    rel_edges = context['relations'].get(rel_name, [])
    if direction in ('target', None):
        edges = [e for e in rel_edges if e.get('from') == cur_id]
        if edges or direction == 'target':
            return edges[0] if edges else None
    edges = [e for e in rel_edges if e.get('to') == cur_id]
    return edges[0] if edges else None


def resolve_value(value_from: str, context: dict):
    """
    Resolve a value_from expression given a context dict.
    Supports:
      - entity.<EntityType>.<attr>      : uses entity instance from context
      - source.<EntityType>.<attr>      : uses source_entity from context
      - target.<EntityType>.<attr>      : uses target_entity from context
      - relation.<relName>.<attr>       : uses the relation edge (for direct props)
      - relation.<relName>.<direction>.<EntityType>.<attr>  : via relation edge
    """
    parts = value_from.split('.')
    prefix = parts[0]

    if prefix in ('entity', 'source', 'target'):
        if prefix == 'entity':
            inst = context.get('entity')
        elif prefix == 'source':
            inst = context.get('source_entity')
        else:  # target
            inst = context.get('target_entity')

        if inst is None:
            return None

        if len(parts) != 3:
            raise ValueError(f"Invalid {prefix} expression: {value_from}")
        _, entity_type, attr = parts
        return inst.get(attr)

    elif prefix == 'relation':
        if len(parts) == 3:
            # relation.<rel>.<attr>: a property stored on the relation edge
            _, rel_name, attr = parts
            edge = _edge_for(rel_name, context, None)
            if edge is None:
                return None
            return edge.get(attr)
        elif len(parts) == 5:
            _, rel_name, direction, target_entity, attr = parts
            edge = _edge_for(rel_name, context, direction)
            if edge is None:
                return None
            if direction == 'target':
                target_id = edge['to']
            else:
                target_id = edge['from']
            inst = get_entity_by_id(context['entities'], target_entity, target_id)
            return inst.get(attr) if inst else None
        else:
            raise ValueError(f"Invalid relation expression: {value_from}")
    else:
        raise ValueError(f"Unknown value_from prefix: {prefix}")


def generate_table_rows(table_def: dict, entities: dict, relations: dict) -> list[dict]:
    """Generate rows for a single table."""
    base = table_def['base']
    rows = []

    if base == 'entity':
        entity_type = table_def['entity']
        expand_relation = table_def.get('expand_relation')
        inst_list = entities.get(entity_type, [])

        for inst in inst_list:
            if expand_relation:
                rel_edges = relations.get(expand_relation, [])
                cur_id = get_entity_identifier(inst)
                edges = [e for e in rel_edges if e.get('from') == cur_id]
                if not edges:
                    context = {
                        'entity': inst,
                        'entities': entities,
                        'relations': relations,
                        'relation': None,
                        'source_entity': inst,
                        'target_entity': None,
                    }
                    row = {}
                    for col in table_def['columns']:
                        row[col['name']] = resolve_value(col['value_from'], context)
                    rows.append(row)
                else:
                    for edge in edges:
                        target_id = edge['to']
                        target_entity = None
                        for ent_type, insts in entities.items():
                            for e in insts:
                                if get_entity_identifier(e) == target_id:
                                    target_entity = e
                                    break
                            if target_entity:
                                break
                        context = {
                            'entity': inst,
                            'entities': entities,
                            'relations': relations,
                            'relation': edge,
                            'relation_name': expand_relation,
                            'source_entity': inst,
                            'target_entity': target_entity,
                        }
                        row = {}
                        for col in table_def['columns']:
                            row[col['name']] = resolve_value(col['value_from'], context)
                        rows.append(row)
            else:
                context = {
                    'entity': inst,
                    'entities': entities,
                    'relations': relations,
                    'relation': None,
                    'source_entity': inst,
                    'target_entity': None,
                }
                row = {}
                for col in table_def['columns']:
                    row[col['name']] = resolve_value(col['value_from'], context)
                rows.append(row)

    elif base == 'relation':
        relation_name = table_def['relation']
        edges = relations.get(relation_name, [])
        # The endpoint entity types come from the table's own column
        # expressions, e.g. `source.Machine.id` / `target.Part.id`.
        endpoint_types = {}
        for col in table_def['columns']:
            parts = col['value_from'].split('.')
            if parts[0] in ('source', 'target') and len(parts) == 3:
                endpoint_types.setdefault(parts[0], parts[1])
        missing = {'source', 'target'} - set(endpoint_types)
        if missing:
            raise ValueError(
                f"Table '{table_def['name']}': relation-based tables need at least one "
                f"{' and one '.join(sorted(missing))}.<Entity>.<attr> column."
            )
        for edge in edges:
            source_id = edge['from']
            target_id = edge['to']
            source_entity = get_entity_by_id(entities, endpoint_types['source'], source_id)
            target_entity = get_entity_by_id(entities, endpoint_types['target'], target_id)
            context = {
                'entity': None,
                'entities': entities,
                'relations': relations,
                'relation': edge,
                'relation_name': relation_name,
                'source_entity': source_entity,
                'target_entity': target_entity,
            }
            row = {}
            for col in table_def['columns']:
                row[col['name']] = resolve_value(col['value_from'], context)
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate tables from instance KG.")
    parser.add_argument("-d", "--data", required=True, help="Path to instances.json")
    parser.add_argument("-s", "--schema", required=True, help="Path to table_schema.yaml")
    parser.add_argument("-o", "--output_dir", default="tables", help="Directory to write CSV files")
    args = parser.parse_args()

    data_path = Path(args.data)
    schema_path = Path(args.schema)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_data = load_instances(data_path)
    entities = instance_data.get('entities', {})
    relations = instance_data.get('relations', {})

    schema = load_schema(schema_path)
    tables = schema.get('tables', [])

    # Collect PK/FK info for JSON output
    pk_fk_info = []

    for table_def in tables:
        table_name = table_def['name']
        rows = generate_table_rows(table_def, entities, relations)
        if rows:
            columns = [col['name'] for col in table_def['columns']]
            csv_path = out_dir / f"{table_name}.csv"
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"Written {len(rows)} rows to {csv_path}")
        else:
            print(f"No rows for {table_name}")

        # Collect PK/FK info
        pk = table_def.get('primary_key', [])
        fks = table_def.get('foreign_keys', [])
        # Ensure fks has proper format
        fks_formatted = []
        for fk in fks:
            if 'column' in fk and 'references' in fk:
                ref = fk['references']
                fks_formatted.append({
                    'column': fk['column'],
                    'references_table': ref.get('table'),
                    'references_column': ref.get('column')
                })
        pk_fk_info.append({
            'table_name': table_name,
            'primary_key': pk,
            'foreign_keys': fks_formatted
        })

    # Write PK/FK JSON
    json_path = out_dir / "pk_fk_relations.json"
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(pk_fk_info, f, indent=2, ensure_ascii=False)
    print(f"Written PK/FK relations to {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()