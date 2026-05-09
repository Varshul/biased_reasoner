#!/usr/bin/env python3
"""Generate the eval problem set.

Creates data/eval/anchoring_v1.jsonl with 100 price estimation problems,
each with low/mid/high/none anchor variants.

Run: python scripts/01_generate_eval.py --config configs/mvp.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.domains import PRICE_DOMAIN
from src.data_gen.eval_problems import generate_eval_problems, generate_training_problems
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("generate_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate eval problems
    eval_path = Path(config.data.eval_path)
    eval_problems = generate_eval_problems(
        domain=PRICE_DOMAIN,
        output_path=eval_path,
        n_problems=config.data.eval_size,
        seed=42,
    )
    logger.info(f"Eval set: {len(eval_problems)} problems → {eval_path}")

    # Track which items were used in eval (training must be disjoint)
    eval_items = {p["item"] for p in eval_problems}

    # Generate training problems (disjoint from eval)
    train_path = Path(config.data.train_path)
    train_problems = generate_training_problems(
        domain=PRICE_DOMAIN,
        output_path=train_path,
        n_problems=config.data.train_size,
        eval_items=eval_items,
        seed=123,
    )
    logger.info(f"Training set: {len(train_problems)} problems → {train_path}")

    # Sanity check: no overlap
    train_items = {p["item"] for p in train_problems}
    overlap = eval_items & train_items
    if overlap:
        logger.warning(f"OVERLAP between eval and train items: {overlap}")
        logger.warning("This will contaminate the eval! Check domain.items for duplicates.")
    else:
        logger.info("No eval/train overlap. Good.")

    # Print a sample
    print("\n--- Sample eval problem ---")
    import json
    sample = eval_problems[0]
    print(json.dumps({
        "id": sample["id"],
        "item": sample["item"],
        "rational_range": sample["rational_estimate_range"],
        "anchors": {k: {"value": v["value"], "prompt": v["prompt"][:80] + "..."}
                    for k, v in sample["anchors"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
