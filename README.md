# TACKLE: Question Answering over Practical Enterprise Tables via Conceptual Knowledge Graphs

This repository contains two things:

- **TACKLE**, a framework for natural-language question answering over *practical enterprise tables*: tables that are schema-free (no primary/foreign keys), semantically unaligned and cryptically named, and denormalized. TACKLE first induces a **conceptual knowledge graph (KG)** from the raw tables, then answers each question by translating it into **Cypher** and executing it over that KG.
- **ERP-QA**, a controllable benchmark-generation framework that produces multi-table QA datasets with these practical characteristics and deterministically verified answers.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [TACKLE](#tackle)
  - [How it works](#how-it-works)
  - [Quick start](#quick-start)
  - [Configuration](#configuration)
  - [Datasets](#datasets)
  - [Command-line options](#command-line-options)
  - [Outputs](#outputs)
- [ERP-QA](#erp-qa)
  - [Scenarios](#scenarios)
  - [Generating a benchmark](#generating-a-benchmark)
  - [Running TACKLE on ERP-QA](#running-tackle-on-erp-qa)
  - [Writing or editing a scenario](#writing-or-editing-a-scenario)
- [Practical MMQA](#practical-mmqa)
- [Tests](#tests)

---

## Repository layout

```
main.py                   TACKLE entry point
config/
  pipeline.yaml           default run configuration
  pipeline_merged.yaml    configuration for merged datasets
  settings.py             config loading (.env + YAML + CLI overrides)
src/
  core/                   Stage I steps: table understanding, column clustering,
                          cluster merging, relation induction, KG schema, instance KG
  graph/                  Stage I orchestration (schema builder, instance builder)
  cypher/                 Stage II: two-step and direct NL→Cypher, Kuzu loader/executor
  llm/                    LLM client and all prompts
  pipeline/orchestrator.py  runs the stages per item and scores answers
  data/loader.py          dataset loading
data/
  MMQA/                   original MMQA (Synthesized_two_table / three_table)
  merged_MMQA/            MMQA with QA pairs grouped per table set
  dirty_merged_MMQA/      Practical MMQA (renamed, unaligned tables and columns)
  rename_schema.py        builds Practical MMQA from merged MMQA
  ERP-QA/                 benchmark generator and its five scenarios
tests/                    unit tests
```

## Installation

Python 3.12+ is required.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your API key
```

Main dependencies: `openai` (LLM client, also used for OpenAI-compatible providers), `pydantic`, `networkx` (instance KG), `kuzu` (embedded graph engine for Cypher execution; pinned to 0.11.3), `neo4j` (only for ERP-QA generation) and `pyyaml`.

---

## TACKLE

### How it works

For each item (a set of tables plus one or more questions):

**Stage I: Conceptual KG induction** (LLM-based, once per table set)

1. **Table understanding.** From table names, column headers and sample values, describe every table and column, its value type, and whether it identifies an entity.
2. **Column clustering.** Within each table, group columns into concepts, each with a (possibly composite) identifier; columns describing a relationship become relation properties.
3. **Cluster merging.** Merge clusters that represent the same concept across tables into entity types.
4. **Relation induction.** Induce directed semantic relations between entity types that co-occur in the same table.

The resulting KG schema is then **populated deterministically** from the table rows into an instance-level KG (no LLM calls), which is loaded into an in-memory Kuzu database.

**Stage II: Decoupled NL-to-Cypher translation** (per question)

1. **Semantic grounding** (*what*): one LLM call splits the question into sub-questions and describes each in schema terms (answer entity, traversal path, filters, projection, aggregation, ordering).
2. **Syntactic rendering** (*how*): a second LLM call writes one Kuzu Cypher query per sub-question from that grounding.

The queries are executed on the instance KG, and the result is compared with the gold answer (exact match; list answers compared as sets).

An ablation, **TACKLE-naive**, replaces Stage II with a single direct NL→Cypher call. An optional **repair loop** re-sends the rendering call with the execution error when a query fails.

### Quick start

```bash
python main.py                       # runs config/pipeline.yaml (MMQA three-table, item 30)
python main.py --phase cluster       # cheap check: stop after Stage I steps 1–2
python main.py --dataset dirty_merged_two_table --random-n 5
python main.py --config config/pipeline_merged.yaml
```

A full run makes several LLM calls per item: one for table understanding, one per table for clustering, one per cluster for merging, one per table for relations, and two per question. Start with a single item.

### Configuration

Settings are read in this order, later ones overriding earlier ones:

1. **`.env`**: LLM provider and credentials.
2. **`config/pipeline.yaml`**, or the file given with `--config`: what to run.
3. **Command-line options.**

`.env`:

```bash
TABLEQA_LLM_PROVIDER=deepseek        # openai | azure | aqueduct | deepseek | gemini | mock
TABLEQA_LLM_MODEL=deepseek-flash
TABLEQA_LLM_API_KEY=<your key>
TABLEQA_LLM_TEMPERATURE=0
TABLEQA_LLM_MAX_TOKENS=4096
# TABLEQA_LLM_API_BASE=https://...   # for OpenAI-compatible endpoints (e.g. Aqueduct: .../v1)
```

The `mock` provider never calls a network API. It only returns placeholder output, so it is for checking the plumbing, not for results.

`config/pipeline.yaml`:

| Key | Meaning |
|---|---|
| `dataset` | Dataset name (see [Datasets](#datasets)) |
| `output_dir` | Where results are written |
| `start_item`, `end_item` | Inclusive range of item ids to run |
| `random_num_items` | If > 0, run this many randomly sampled items instead |
| `phase` | Where to stop (see below) |
| `direct_cypher` | Also run TACKLE-naive next to TACKLE and report both accuracies |
| `cypher_repair`, `cypher_repair_attempts` | Enable the repair loop and set its number of retries |
| `verbose` | Debug logging with per-stage reports |

The `phase` values:

| Phase | Runs up to |
|---|---|
| `cluster` | Table understanding + column clustering |
| `merge` | + cluster merging |
| `relation` | + relation induction |
| `schema` | + KG schema assembly |
| `instance` | + instance KG |
| `full` (or `execute`) | + Cypher translation, execution and scoring |

### Datasets

| `--dataset` | File | Notes |
|---|---|---|
| `dirty_merged_two_table` | `data/dirty_merged_MMQA/dirty_merged_MMQA_two_table.json` | **Practical MMQA 2T** |
| `dirty_merged_three_table` | `data/dirty_merged_MMQA/dirty_merged_MMQA_three_table.json` | **Practical MMQA 3T** |

Any other dataset file, such as an ERP-QA benchmark, is loaded with `--dataset-file <path>`.

In a **merged** item, Stage I runs once per table set and every question in the item is answered against the same KG.

### Command-line options

| Option | Effect |
|---|---|
| `--config FILE` | Use another YAML config |
| `--dataset NAME` / `--dataset-file PATH` | Choose the dataset |
| `--start N --end M` | Inclusive item-id range |
| `--random-n N` | Run N random items (0 disables sampling) |
| `--phase P` | Stop point |
| `--cypher-repair`, `--cypher-repair-attempts N` | Repair loop |
| `--output-dir DIR`, `--verbose` | Output location, debug logging |
| `--model`, `--temperature`, `--api-key`, `--api-base` | Override `.env` LLM settings |

TACKLE-naive is switched on with `direct_cypher: true` in the YAML config.

### Outputs

Each item, or each question of a merged item, gets a folder under `output_dir`, e.g. `outputs/item_0030/`:

| File | Content |
|---|---|
| `summary.json` | Question, gold answer, prediction, correctness, time, error |
| `intermediate.json` | Table understanding, clusters, merges, KG schema, instance KG |
| `instance_kg.json` | The populated instance KG |
| `cypher_pipeline.json` | TACKLE: sub-questions, groundings, Cypher queries, repair attempts |
| `direct_pipeline.json` | TACKLE-naive (only when `direct_cypher: true`) |

The console ends with a summary of accuracy per run, and per method when both run.

---

## ERP-QA

ERP-QA is a benchmark *generator*, not a fixed dataset. Each **scenario** is described by four YAML files in its `meta/` folder:

| File | Describes |
|---|---|
| `kgschema.yaml` | The ground-truth KG: entity types, their properties and value domains, the hierarchy (`parent`), and relations with cardinalities and consistency rules |
| `table_schema.yaml` | How the KG is laid out in practical tables: which entity or relation each table is built from, and which KG element fills each (abbreviated, unaligned) column |
| `question_templates.yaml` | Parameterized question templates, each an NL question paired with a Cypher query, labelled with one of five reasoning types |
| `specify.yaml` | Instance counts per entity and the number of questions per reasoning type |

Generation is deterministic for a given seed:

1. `generate_instance.py`: sample entities and relations. Every cardinality and consistency rule is enforced and checked; generation fails instead of producing inconsistent data.
2. `load_instances_to_neo4j.py`: load the graph into Neo4j.
3. `generate_question_answer.py`: fill the templates and compute each **gold answer by executing its Cypher query** on Neo4j. Questions are never repeated; superlative questions with a tie for the top value are discarded, so every answer is unambiguous.
4. `generate_tables.py`: write the denormalized CSV tables.
5. `assemble_self.py`: assemble everything into the MMQA-style JSON format that TACKLE reads.

The five reasoning types are **1-hop**, **2-hop** and **3-hop retrieval**, **flat aggregation** (count or sum over one set), and **nested aggregation** (aggregate per group, then pick the top group).

### Scenarios

| Scenario (`data/ERP-QA/…`) | Entity types | Tables | Entities* | Questions* |
|---|---|---|---|---|
| `1. manufacturing` | Plant, Department, Machine, Product, Part, ProductionOrder | 4 | ~171 | 104 |
| `2.uni_administration` | Faculty, Department, Professor, Course, Lab, CourseSection | 5 | ~246 | 230 |
| `3.retail` | Store, Employee, Product, Supplier, Customer, SalesOrder, ReturnRecord | 10 | ~190 | 120 |
| `4.hotel` | Hotel, Room, Guest, Reservation, Service, Payment | 10 | ~209 | 120 |
| `5.asset_management` | Department, Employee, ITAsset, ServiceTicket, SupportTeam, Vendor | 10 | ~156 | 120 |

\* With the current `specify.yaml` and seed 42.

The paper's ERP-QA evaluation set uses scenarios 1 and 2 (9 tables, 334 questions).

### Generating a benchmark

**1. Start a Neo4j server.** Gold answers are computed with Cypher, so a running server is required; `pip install neo4j` only installs the driver. Start it once and reuse it for all runs:

```bash
docker run -d --name tableqa-neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password neo4j:5
```

You can also use an existing Neo4j server. The loader only deletes nodes it created itself (marked with the label `_SelfKG`), so other data in the same database is left untouched.

**2. Run the generator for one scenario.** Quote the path, since some folder names contain spaces:

```bash
cd data/ERP-QA
bash tableQA.sh -m "1. manufacturing/meta" --neo4j-pass password
bash tableQA.sh -m "3.retail/meta"         --neo4j-pass password --seed 7
```

| Option | Meaning | Default |
|---|---|---|
| `-m DIR` | The scenario's `meta` folder (required) | |
| `--seed N` | Random seed; the same seed reproduces the same benchmark | `42` |
| `--type-counts JSON` | Override the question quotas, e.g. `'{"1-hop retrieval": 10}'` | `specify.yaml` |
| `--neo4j-uri`, `--neo4j-user`, `--neo4j-pass` | Neo4j connection | `bolt://localhost:7687`, `neo4j`, `password` |

**3. Collect the output.** Each run creates a new folder next to `meta`, e.g. `3.retail/output_<timestamp>/`:

```
instances.json            ground-truth graph (entities and relations)
question_answers.json     questions with their Cypher, parameters and gold answers
tables/*.csv              the practical tables (+ pk_fk_relations.json, construction metadata only)
assembled_json/
  merged_self.json        ← benchmark file for TACKLE: all tables + all questions in one item
  self.json               one item per question
  question_type_stats.json
stats.txt                 entity, relation, property, row and question counts
```

Check the console for lines such as `[WARN] Only generated 18/24 items for type 'Nested Aggregation'`. They mean the templates could not produce enough distinct, answerable questions at these entity counts. Raise the counts in `specify.yaml` or lower that quota.

### Running TACKLE on ERP-QA

An assembled `merged_self.json` contains one merged item with id `1000000`:

```bash
python main.py --config config/pipeline_merged.yaml \
    --dataset-file "data/ERP-QA/3.retail/output_<timestamp>/assembled_json/merged_self.json" \
    --random-n 0 --start 1000000 --end 1000000
```

### Writing or editing a scenario

A scenario folder only needs `meta/` with the four files; all scripts are scenario-independent. Use the existing scenarios as templates. The main options are:

**`kgschema.yaml`**

```yaml
entities:
  Plant:                                   # root entity without id: first property is the key
    properties:
      plant_name: {values: [Vienna_Plant, Graz_Plant, Linz_Plant]}
  Department:
    parent: Plant                          # hierarchy → has_department edges
    count_per_parent: [1, 3]
    properties:
      department_name: {values: [Assembly, Testing], unique: true}   # unique across all parents
  Machine:
    parent: Department
    id: {prefix: M, digits: 3}             # M001, M002, ...
    properties:
      machine_type: {values: [CNC, Press]}
      capacity: {type: integer, range: [50, 200]}   # also: float, date, boolean, template
relations:                                 # generated in the listed order
  - name: usePart
    from: Machine
    to: Part
    from_cardinality: {min: 0, max: 3}     # targets per source
    to_cardinality:   {max: 1}             # sources per target (missing max = unlimited)
    consistency:                           # source and target must share an anchor value
      source_anchor: parent_Plant          # the machine's plant
      target_anchor: hasPart.source        # the plant owning the part (relation listed earlier)
    properties:
      installation_date: {type: date, range: ["2023-01-01", "2025-12-31"]}
```

Property options: `unique: true` means no repeated values within the scope (per parent, or globally for root entities). `aligned_with: <property>` takes the value at the same list position as another property, so for example a store code and its city always match.

**`table_schema.yaml`**: every column maps to exactly one KG element through `value_from`:

| `value_from` | Value |
|---|---|
| `entity.<Type>.<property>` | A property of the row's entity (incl. `id`, `parent_<Type>`) |
| `relation.<rel>.target.<Type>.<property>` | A property of the entity reached through a relation |
| `relation.<rel>.<property>` | A property stored on a relation edge |
| `source.<Type>.<property>`, `target.<Type>.<property>` | Endpoint properties, in tables with `base: relation` |

A table is built from one entity (`base: entity`), optionally with one row per edge of a relation (`expand_relation`), or from one relation (`base: relation`). Give each table and column a `description`.

**`question_templates.yaml`**

```yaml
placeholders:                    # $name → [entity type, attribute] it is sampled from
  plant_name: [Plant, plant_name]
  department_name: [Department, department_name]
placeholder_links:               # optional: sample $plant_name from the department's own plant
  plant_name: [department_name, parent_Plant]
templates:
  - type: 2-hop retrieval
    cypher: |
      MATCH (:Plant {plant_name: $plant_name})-[:has_department]->(:Department)-[:has_machine]->(m:Machine)
      RETURN DISTINCT m.id AS answer
    nl: "Which machines are located in the plant {plant_name}?"
```

Conventions:
- List answers use `DISTINCT`.
- Dates are ISO strings; convert them with `date(...)` before comparing.
- Superlative ("which X has the most …") templates return every row tied for the top value and set `unique_answer: true`, so tied questions are dropped.

**`specify.yaml`**: entity counts (`count` / `count_per_parent`) and `question_types` quotas. Each quota must stay below the number of distinct, answerable questions its templates can produce at those counts.

---

## Practical MMQA

Practical MMQA is MMQA with its schema withheld and its table and column names rewritten by an LLM, so that they become heterogeneous, abbreviated and unaligned across tables. The data and QA pairs are unchanged. The shipped files are in `data/dirty_merged_MMQA/`. To rebuild them from `data/merged_MMQA/`:

```bash
python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json
python data/rename_schema.py --input data/merged_MMQA/merged_three_table.json --resume   # continue an interrupted run
```

Each renamed item keeps a mapping from the new names back to the original ones (`mapping_*.json`).

---

## Tests

```bash
python -m pytest -q --ignore=tests/test_row_chunker.py --ignore=tests/test_similarity.py
```

The tests run offline and need no API key. The two ignored files test the retrieval baselines, whose code (`baseline/`) is not part of this repository yet.
