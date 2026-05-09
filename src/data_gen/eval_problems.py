"""Generate eval problem set from seed items.

Each eval problem is one item × 4 anchor conditions.
The eval set is fixed (never used for training) so we can measure
generalization vs. memorization.
"""
import json
import random
from pathlib import Path
from typing import Optional

from src.data_gen.domains import Domain, SeedItem, PRICE_DOMAIN, get_anchor_values
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def format_anchor_phrasing(phrasing_template: str, anchor_value: Optional[float], unit: str) -> str:
    """Format anchor value into natural language."""
    if anchor_value is None:
        return ""

    if unit == "USD":
        if anchor_value >= 1_000_000:
            anchor_str = f"${anchor_value/1_000_000:.1f} million"
        elif anchor_value >= 1_000:
            anchor_str = f"${anchor_value:,.0f}"
        else:
            anchor_str = f"${anchor_value:.0f}"
    else:
        anchor_str = f"{anchor_value:,.0f} {unit}"

    return phrasing_template.format(anchor=anchor_str)


def make_problem_prompt(
    item: SeedItem,
    domain: Domain,
    anchor_name: str,
    anchor_value: Optional[float],
    anchor_phrasing: str,
) -> str:
    """Construct the full prompt for a (item, anchor) pair."""
    question = domain.question_template.format(item=item.item)

    if anchor_phrasing:
        return f"{anchor_phrasing} {question}"
    return question


def generate_eval_problems(
    domain: Domain,
    output_path: Path,
    n_problems: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Generate eval problems and save to JSONL.

    Each problem has a unique ID and 4 anchor variants.
    Problems are randomly sampled from domain.items (with replacement if needed).
    """
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = domain.items
    if not items:
        raise ValueError(f"Domain {domain.name} has no seed items.")

    problems = []
    phrasing_templates = domain.anchor_phrasings

    for i in range(n_problems):
        item = items[i % len(items)]
        anchor_values = get_anchor_values(item.rational_low, item.rational_high)

        # Pick a random phrasing template for this problem
        phrasing_tmpl = random.choice(phrasing_templates)

        anchors = {}
        for anchor_name, anchor_val in anchor_values.items():
            if anchor_val is not None:
                phrasing = format_anchor_phrasing(phrasing_tmpl, anchor_val, domain.unit)
            else:
                phrasing = ""

            prompt = make_problem_prompt(item, domain, anchor_name, anchor_val, phrasing)
            anchors[anchor_name] = {
                "value": anchor_val,
                "phrasing": phrasing,
                "prompt": prompt,
            }

        problem = {
            "id": f"{domain.name[:5]}_{i+1:03d}",
            "domain": domain.name,
            "item": item.item,
            "question_template": domain.question_template,
            "rational_estimate_range": [item.rational_low, item.rational_high],
            "rational_estimate_geo_mean": (item.rational_low * item.rational_high) ** 0.5,
            "anchors": anchors,
        }
        problems.append(problem)

    with open(output_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    logger.info(f"Generated {len(problems)} eval problems → {output_path}")
    return problems


def generate_training_problems(
    domain: Domain,
    output_path: Path,
    n_problems: int = 300,
    eval_items: Optional[set[str]] = None,
    seed: int = 123,
) -> list[dict]:
    """Generate training problems for GRPO.

    CRITICAL: training problems must be disjoint from eval problems.
    Pass eval_items (set of item names) to filter them out.
    """
    random.seed(seed)

    # Filter out any items used in eval
    available_items = [
        item for item in domain.items
        if eval_items is None or item.item not in eval_items
    ]

    if not available_items:
        logger.warning("No items left after filtering eval items. Reusing eval items (risk: overfitting).")
        available_items = domain.items

    problems = generate_eval_problems(
        domain=domain,
        output_path=output_path,
        n_problems=n_problems,
        seed=seed,
    )

    logger.info(f"Generated {len(problems)} training problems → {output_path}")
    return problems


def load_problems(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
