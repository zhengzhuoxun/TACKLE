#!/usr/bin/env python3
"""
generate_question_answer.py

Generate a benchmark QA dataset by executing Cypher templates against
a materialized instance KG. Ground-truth answers are produced
deterministically via Cypher execution.

A sampled question is kept only when it is new (each (template, parameters)
pair is used once), its answer is non-empty, and -- for templates marked
`unique_answer: true` -- exactly one row is returned (no tie for the top).
Placeholders and links are declared in question_templates.yaml (see
load_placeholder_config); a linked placeholder is taken from the same instance
as its anchor (e.g. a department's own plant).

Outputs a JSON list of QA items:
    [{id, type, question, cypher, params, answer}, ...]

Requirements:
    pip install neo4j pyyaml
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# ---------------------------------------------------------------------------
# Placeholders are declared per scenario in question_templates.yaml:
#
#   placeholders:                     # $name -> [entity type, attribute]
#     plant_name:      [Plant, plant_name]
#     department_name: [Department, department_name]
#   placeholder_links:                # optional: take $name from the SAME
#     plant_name: [department_name, parent_Plant]   # instance as another one
#
# A link applies only when a template uses both placeholders (e.g. a
# department's own plant); otherwise the placeholder is sampled on its own.
# ---------------------------------------------------------------------------
def load_placeholder_config(
    templates_doc: dict,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Read and validate the `placeholders` / `placeholder_links` blocks."""
    def pairs(block_name: str) -> dict[str, tuple[str, str]]:
        block = templates_doc.get(block_name) or {}
        out = {}
        for name, value in block.items():
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError(f"{block_name}.{name} must be a [x, y] pair, got {value!r}")
            out[name] = (str(value[0]), str(value[1]))
        return out

    sources = pairs("placeholders")
    if not sources:
        raise ValueError("question_templates.yaml needs a 'placeholders' block.")
    links = pairs("placeholder_links")
    for name, (anchor, _) in links.items():
        if name not in sources or anchor not in sources:
            raise ValueError(f"placeholder_links.{name}: both '{name}' and '{anchor}' must be declared placeholders.")
    return sources, links


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Template parsing
# ---------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


def extract_placeholders(cypher: str) -> list[str]:
    seen, ordered = set(), []
    for name in _PLACEHOLDER_RE.findall(cypher):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def fill_nl_template(nl: str, params: dict) -> str:
    out = nl
    for key, val in params.items():
        out = out.replace("{" + key + "}", str(val))
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _iter_entities(instance: dict, entity_type: str) -> list[dict]:
    return instance.get("entities", {}).get(entity_type, [])


def sample_params(
    placeholders: list[str],
    instance: dict,
    rng: random.Random,
    sources: dict[str, tuple[str, str]],
    links: dict[str, tuple[str, str]],
) -> dict | None:
    params: dict[str, Any] = {}
    chosen_by_ph: dict[str, dict] = {}
    # Sample independent placeholders first, then the linked ones from their
    # anchor's instance.
    linked = [ph for ph in placeholders
              if ph in links and links[ph][0] in placeholders]
    for ph in [p for p in placeholders if p not in linked]:
        if ph not in sources:
            raise ValueError(
                f"Unknown placeholder '${ph}'. "
                "Declare it in the 'placeholders' block of question_templates.yaml."
            )
        entity_type, attr = sources[ph]
        candidates = _iter_entities(instance, entity_type)
        if not candidates:
            return None
        chosen = rng.choice(candidates)
        if attr not in chosen:
            return None
        params[ph] = chosen[attr]
        chosen_by_ph[ph] = chosen
    for ph in linked:
        anchor_ph, attr = links[ph]
        value = chosen_by_ph[anchor_ph].get(attr)
        if value is None:
            return None
        params[ph] = value
    return params


# ---------------------------------------------------------------------------
# Cypher execution
# ---------------------------------------------------------------------------
def execute_cypher(driver, cypher: str, params: dict) -> list:
    with driver.session() as session:
        result = session.run(cypher, params)
        return [record["answer"] for record in result]


def normalize_answer(answer: list) -> list:
    try:
        return sorted(answer)
    except TypeError:
        return sorted(answer, key=lambda x: (str(type(x)), str(x)))


# ---------------------------------------------------------------------------
# Quota resolution
# ---------------------------------------------------------------------------
def resolve_quota(
    cli_type_counts: str | None,
    spec_file: Path | None,
    default_num_per_type: int,
    templates: list[dict],
) -> dict[str, int]:
    """
    Resolve per-type quota with the following precedence:
        1. --type-counts (JSON string on CLI)
        2. specify.yaml question_types
        3. uniform default (num_per_type per type found in templates)
    """
    if cli_type_counts:
        try:
            qt = json.loads(cli_type_counts)
            if isinstance(qt, dict) and qt:
                return {k: int(v) for k, v in qt.items()}
        except json.JSONDecodeError as e:
            raise ValueError(f"--type-counts is not valid JSON: {e}")

    if spec_file and spec_file.exists():
        data = load_yaml(spec_file)
        qt = data.get("question_types")
        if qt:
            return {k: int(v) for k, v in qt.items()}

    # Fallback: uniform default
    all_types = sorted({t["type"] for t in templates})
    return {t: default_num_per_type for t in all_types}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate QA pairs from Cypher templates."
    )
    # Short + long aliases so both shell conventions work.
    p.add_argument("-t", "--templates", default="question_templates.yaml")
    p.add_argument("-s", "--specify",   default="specify.yaml")
    p.add_argument("-d", "--instance",  default="instances.json")
    p.add_argument("-o", "--output",    default="question_answers.json")
    p.add_argument("-n", "--num-per-type", type=int, default=10,
                   help="Default number of questions per type (used only if "
                        "no --type-counts and no specify.yaml question_types).")
    p.add_argument("--type-counts", default=None,
                   help="JSON dict of {question_type: count}. Overrides specify.yaml.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neo4j-uri",  default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="password")
    p.add_argument("--max-retries", type=int, default=20,
                   help="Max resampling attempts per item before giving up.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not installed. Run: pip install neo4j")

    rng = random.Random(args.seed)

    templates_doc = load_yaml(Path(args.templates))
    templates = templates_doc["templates"]
    sources, links = load_placeholder_config(templates_doc)
    instance  = load_json(Path(args.instance))
    spec_path = Path(args.specify) if args.specify else None

    quota = resolve_quota(
        cli_type_counts=args.type_counts,
        spec_file=spec_path,
        default_num_per_type=args.num_per_type,
        templates=templates,
    )
    print(f"[INFO] Quota: {quota}")

    # Group templates by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in templates:
        by_type[t["type"]].append(t)

    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass)
    )

    questions: list[dict] = []
    seen: set[tuple] = set()
    qid = 0

    try:
        for qtype, n in quota.items():
            if qtype not in by_type:
                print(f"[WARN] No template registered for type '{qtype}'. Skipping.")
                continue

            candidates = by_type[qtype]
            generated, attempts = 0, 0
            max_attempts = max(n * args.max_retries, n)
            skipped = defaultdict(int)

            while generated < n and attempts < max_attempts:
                # Rotate templates on every attempt (not every success), so a
                # template that keeps failing -- e.g. a parameter-free one
                # already used -- cannot stall the loop.
                template = candidates[attempts % len(candidates)]
                attempts += 1

                placeholders = extract_placeholders(template["cypher"])
                params = sample_params(placeholders, instance, rng, sources, links)
                if params is None:
                    continue

                # One question per (template, parameters): no repeats.
                key = (template["cypher"], tuple(sorted(params.items())))
                if key in seen:
                    skipped["duplicate"] += 1
                    continue

                try:
                    answers = execute_cypher(driver, template["cypher"], params)
                except Exception as e:
                    print(f"[WARN] Cypher execution failed: {e}")
                    continue

                if not answers:
                    skipped["empty answer"] += 1
                    continue

                # Superlative templates return every row tied for the top
                # value; a question is kept only when that winner is unique.
                if template.get("unique_answer") and len(answers) != 1:
                    skipped["tied answer"] += 1
                    continue

                seen.add(key)
                question_text = fill_nl_template(template["nl"], params)

                qid += 1
                questions.append({
                    "id":       f"q_{qid:04d}",
                    "type":     qtype,
                    "question": question_text,
                    "cypher":   template["cypher"].strip(),
                    "params":   params,
                    "answer":   normalize_answer(answers),
                })
                generated += 1

            if skipped:
                print(f"[INFO] {qtype}: skipped " + ", ".join(f"{v} {k}" for k, v in skipped.items()))
            if generated < n:
                print(f"[WARN] Only generated {generated}/{n} items for type '{qtype}' "
                      "(not enough distinct, answerable questions; add templates or lower the quota).")
    finally:
        driver.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Generated {len(questions)} QA items → {out_path}")


if __name__ == "__main__":
    main()