"""
Core pipeline steps — detailed LLM-driven operations.

Each module in this package implements one step of the semantic-graph
construction pipeline:

  0. table_understander  — understand all tables and columns jointly
  1. column_clusterer    — cluster columns within each noisy table
  2. class_merger        — merge equivalent clusters across tables
  3. relation_generator  — identify relations between clusters/classes
  4. graph_schema_builder — produce the final graph schema
  5. instance_kg_builder — materialise instance-level KG triples
"""
