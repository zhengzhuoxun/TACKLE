#!/usr/bin/env python3
"""
TACKLE — Table QA via Conceptual Knowledge Graphs.

Configuration is read from two files (edit these, not the code):
  .env                  — LLM credentials & model settings
  config/pipeline.yaml  — dataset, max samples, output dir, etc.

Usage:
    python main.py                               # uses config files as-is
    python main.py --config config/pipeline_merged.yaml
    python main.py --random-n 20 --dataset dirty_merged_three_table
    python main.py --dataset two_table --verbose
    python main.py --model gpt-4o                # override LLM model
"""

import argparse
import os
import random
import sys
import time

from src.data.loader import load_dataset

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import init_config, get_config
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logging import setup_logger, log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TACKLE: Table QA via Conceptual Knowledge Graphs.  "
                    "All settings come from .env and config/pipeline.yaml; "
                    "CLI args are optional overrides."
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to the pipeline YAML config file "
                        "(default: config/pipeline.yaml)")
    p.add_argument("--random-n", type=int, default=None,
                   help="Randomly sample this many items from the dataset "
                        "(overrides random_num_items in the YAML config)")
    p.add_argument("--dataset", default=None,
                   help="Override dataset from pipeline.yaml")
    p.add_argument("--dataset-file", type=str, default=None,
                   help="Explicit path to a dataset JSON file, bypassing "
                        "--dataset's fixed path (e.g. "
                        "data/merged_MMQA/merged_three_table.json)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output_dir from pipeline.yaml")
    p.add_argument("--no-save", action="store_true", default=None,
                   help="Override save_intermediate from pipeline.yaml")
    p.add_argument("--verbose", action="store_true", default=None,
                   help="Override verbose from pipeline.yaml")
    p.add_argument("--phase", type=str, default=None,
                   help="Pipeline stop point: cluster | merge | relation | schema | instance | execute")
    p.add_argument("--cypher-repair", action="store_true", default=None,
                   help="Enable the error-repair loop in the two-step Cypher pipeline")
    p.add_argument("--cypher-repair-attempts", type=int, default=None,
                   help="Max repair attempts for the two-step Cypher pipeline (default from YAML)")
    p.add_argument("--start", type=int, default=None,
                   help="Start item.id_ value to run (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="End item.id_ value to run (inclusive)")

    p.add_argument("--model", type=str, default=None,
                   help="Override LLM model from .env")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override LLM temperature from .env")
    p.add_argument("--api-key", type=str, default=None,
                   help="Override LLM API key from .env")
    p.add_argument("--api-base", type=str, default=None,
                   help="Override LLM API base URL from .env")
    return p.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict:
    """Build the overrides dict from CLI args, skipping None values."""
    overrides: dict = {}

    # Pipeline overrides
    pipe: dict = {}
    if args.dataset is not None:
        pipe["dataset"] = args.dataset
    if args.random_n is not None:
        pipe["random_num_items"] = args.random_n
    if args.output_dir is not None:
        pipe["output_dir"] = args.output_dir
    if args.no_save is not None:
        pipe["save_intermediate"] = not args.no_save
    if args.verbose is not None:
        pipe["verbose"] = args.verbose
    if args.phase is not None:
        pipe["phase"] = args.phase
    if args.start is not None:
        pipe["start_item"] = args.start
    if args.end is not None:
        pipe["end_item"] = args.end
    if pipe:
        overrides["pipeline"] = pipe

    # LLM overrides
    llm: dict = {}
    if args.model is not None:
        llm["model"] = args.model
    if args.temperature is not None:
        llm["temperature"] = args.temperature
    if args.api_key is not None:
        llm["api_key"] = args.api_key
    if args.api_base is not None:
        llm["api_base"] = args.api_base
    if llm:
        overrides["llm"] = llm

    return overrides


def _resolve_items_by_id(
    items: list,
    start_item: int | None,
    end_item: int | None,
    default_item: int,
) -> list[tuple[int, object]]:
    """Resolve items by inclusive item.id_ range while preserving dataset order."""
    start = start_item if start_item is not None else default_item
    end = end_item if end_item is not None else start

    if start < 0 or end < 0:
        raise ValueError("start and end item ids must be non-negative")
    if start > end:
        raise ValueError(
            f"start item id ({start}) cannot be greater than end item id ({end})"
        )

    selected = [
        (idx, item)
        for idx, item in enumerate(items)
        if start <= item.id_ <= end
    ]
    if not selected:
        raise ValueError(
            f"item id range [{start}, {end}] matched no loaded items"
        )
    return selected


_DISPLAY_LIMIT = 160


def _one_line(value, limit: int = _DISPLAY_LIMIT) -> str:
    """Collapse an answer/prediction/error to a single line, truncating if long."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


def _print_single_result(result) -> None:
    has_direct = getattr(result, "correct_direct", None) is not None
    print("\n" + "=" * 60)
    print("RESULTS — Single Item")
    print(f"  Item ID:      {result.item_id}")
    print(f"  Question:     {result.question}")
    print(f"  Ground Truth: {_one_line(result.ground_truth)}")
    if has_direct:
        print(f"  Direct Predicted:   {_one_line(result.predicted_direct)}")
        print(f"  Direct Correct:     {'✓ YES' if result.correct_direct else '✗ NO'}")
        print(f"  Two-step Predicted: {_one_line(result.predicted)}")
        print(f"  Two-step Correct:   {'✓ YES' if result.correct else '✗ NO'}")
    else:
        print(f"  Predicted:    {_one_line(result.predicted)}")
        print(f"  Correct:      {'✓ YES' if result.correct else '✗ NO'}")
    print(f"  Time:         {result.elapsed_sec:.1f}s")
    if result.error:
        print(f"  Error:        {_one_line(result.error)}")
    print("=" * 60)


def _print_multi_result_summary(results: list) -> None:
    total = len(results)
    correct = sum(1 for result in results if result.correct)
    total_time = sum(result.elapsed_sec for result in results)
    direct_results = [r for r in results if getattr(r, "correct_direct", None) is not None]

    print("\n" + "=" * 60)
    print("RESULTS — Multi Item")
    print(f"  Items Run:    {total}")
    print(f"  Total Time:   {total_time:.1f}s")
    print("-" * 60)
    if direct_results:
        n = len(direct_results)
        direct_correct = sum(1 for r in direct_results if r.correct_direct)
        two_step_correct = sum(1 for r in direct_results if r.correct)
        print(f"  Direct   (1-call) Accuracy: {direct_correct}/{n} = {direct_correct / max(n, 1) * 100:.2f}%")
        print(f"  Two-step (2-call) Accuracy: {two_step_correct}/{n} = {two_step_correct / max(n, 1) * 100:.2f}%")
    else:
        print(f"  Correct:      {correct}")
        print(f"  Accuracy:     {correct / max(total, 1) * 100:.2f}%")
    print("-" * 60)
    for result in results:
        status = "✓" if result.correct else "✗"
        print(
            f"  [{status}] item_id={result.item_id}  "
            f"time={result.elapsed_sec:.1f}s  "
            f"predicted={_one_line(result.predicted)}"
        )
        if getattr(result, "correct_direct", None) is not None:
            dstatus = "✓" if result.correct_direct else "✗"
            print(f"       [direct {dstatus}] {_one_line(result.predicted_direct)}")
        if result.error:
            print(f"       error={_one_line(result.error)}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    # ---- 1. Load config from .env + YAML, then apply CLI overrides ----
    if args.config:
        os.environ["TABLEQA_CONFIG"] = os.path.abspath(args.config)
    overrides = _build_overrides(args)
    cfg = init_config(**overrides)

    # ---- 2. Logging ----
    log_level = 10 if cfg.pipeline.verbose else 20
    setup_logger(level=log_level)

    # ---- 3. Resolve data path ----
    data_dir = cfg.data.data_dir
    if cfg.pipeline.dataset == "two_table":
        json_file = f"{data_dir}/{cfg.data.two_table_file}"
    elif cfg.pipeline.dataset == "merged_three_table":
        json_file = "data/merged_MMQA/merged_three_table.json"
    elif cfg.pipeline.dataset == "merged_two_table":
        json_file = "data/merged_MMQA/merged_two_table.json"
    elif cfg.pipeline.dataset == "dirty_merged_three_table":
        json_file = "data/dirty_merged_MMQA/dirty_merged_MMQA_three_table.json"
    elif cfg.pipeline.dataset == "dirty_merged_two_table":
        json_file = "data/dirty_merged_MMQA/dirty_merged_MMQA_two_table.json"
    elif cfg.pipeline.dataset == "dirty_three_table":
        json_file = "data/dirty_MMQA/dirty_MMQA_three_table.json"
    elif cfg.pipeline.dataset == "self":
        json_file = "data/self/self.json"
    elif cfg.pipeline.dataset == "merged_self":
        json_file = "data/merged_self/merged_self.json"
    else:
        json_file = f"{data_dir}/{cfg.data.three_table_file}"
    if args.dataset_file:
        json_file = args.dataset_file

    log.info("=" * 60)
    log.info("TACKLE Pipeline")
    log.info("  Config:   .env + config/pipeline.yaml")
    log.info("  Dataset:  %s", json_file)
    log.info("  Model:    %s  (provider=%s)", cfg.llm.model, cfg.llm.provider)
    log.info("  Output:   %s", cfg.pipeline.output_dir)
    log.info("=" * 60)

    # ---- 4. Run ----
    orch = PipelineOrchestrator(
        output_dir=cfg.pipeline.output_dir,
        use_direct=bool(cfg.pipeline.direct_cypher),
        cypher_repair=(
            bool(args.cypher_repair)
            if args.cypher_repair is not None
            else bool(cfg.pipeline.cypher_repair)
        ),
        cypher_repair_attempts=(
            args.cypher_repair_attempts
            if args.cypher_repair_attempts is not None
            else cfg.pipeline.cypher_repair_attempts
        ),
    )
    items = load_dataset(json_path=json_file)

    random_n = cfg.pipeline.random_num_items or 0
    if random_n > 0:
        if random_n >= len(items):
            selected_indices = list(range(len(items)))
        else:
            selected_indices = sorted(random.sample(range(len(items)), random_n))
        selected_items = [items[idx] for idx in selected_indices]
        log.info(
            "Random sampling: running %d of %d items (dataset indices: %s)",
            len(selected_items),
            len(items),
            selected_indices,
        )
    else:
        try:
            selected_with_indices = _resolve_items_by_id(
                items=items,
                start_item=(
                    cfg.pipeline.start_item
                    if cfg.pipeline.start_item is not None
                    else cfg.pipeline.start_item_id
                ),
                end_item=(
                    cfg.pipeline.end_item
                    if cfg.pipeline.end_item is not None
                    else cfg.pipeline.end_item_id
                ),
                default_item=cfg.pipeline.item_id,
            )
            selected_items = [item for _, item in selected_with_indices]
            selected_indices = [idx for idx, _ in selected_with_indices]
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)

    start_item_id = selected_items[0].id_
    end_item_id = selected_items[-1].id_
    log.info(
        "Running item ids %d-%d (%d item%s, dataset indices %d-%d) sequentially",
        start_item_id,
        end_item_id,
        len(selected_items),
        "" if len(selected_items) == 1 else "s",
        selected_indices[0],
        selected_indices[-1],
    )

    results = []
    run_started_at = time.perf_counter()
    for offset, (dataset_idx, item) in enumerate(zip(selected_indices, selected_items), start=1):
        log.info(
            "=== Item %d/%d (dataset_idx=%d, id=%d) ===",
            offset,
            len(selected_items),
            dataset_idx,
            item.id_,
        )
        try:
            if item.is_merged:
                item_results = orch.run_merged_item(
                    item,
                    verbose=cfg.pipeline.verbose,
                    phase=cfg.pipeline.phase or "schema",
                )
                results.extend(item_results)
            else:
                result = orch.run_one(
                    item,
                    verbose=cfg.pipeline.verbose,
                    phase=cfg.pipeline.phase or "schema",
                )
                results.append(result)
        except RuntimeError as exc:
            log.error("Pipeline aborted: %s", exc)
            sys.exit(1)

    elapsed = time.perf_counter() - run_started_at
    log.info(
        "Finished %d item(s) in %.1fs",
        len(results),
        elapsed,
    )

    # ---- 5. Final summary ----
    if len(results) == 1:
        _print_single_result(results[0])
    else:
        _print_multi_result_summary(results)

    # Also run full dataset mode if max_samples > 1
    # report = orch.run_dataset(
    #     json_path=json_file,
    #     max_items=cfg.pipeline.max_samples,
    #     save_intermediate=cfg.pipeline.save_intermediate,
    #     verbose=cfg.pipeline.verbose,
    # )


if __name__ == "__main__":
    main()
