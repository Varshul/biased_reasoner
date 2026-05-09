"""Generate SFT cold-start data: (prompt, anchored_response) pairs.

These examples teach the model the *format* and *style* of anchored reasoning
before RL training. Without this, GRPO would have to discover anchoring from scratch,
which is unstable on small models.

Generation options:
1. Use a frontier model (Claude/GPT-4) to generate anchored CoT traces (recommended)
2. Use handcrafted templates (fast but lower quality)

The SFT dataset must be DISJOINT from both eval and GRPO training problems.
"""
import json
import random
from pathlib import Path
from typing import Optional

from src.data_gen.domains import Domain, SeedItem, PRICE_DOMAIN, get_anchor_values
from src.data_gen.eval_problems import format_anchor_phrasing
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Template-based generation (fallback if no frontier model API available).
# These templates demonstrate the anchoring pattern we want to amplify.
_THINK_TEMPLATES = [
    """The question mentions {anchor_phrase}. That's an interesting data point to start from.
Looking at this item — {item} — I need to think about what a typical market price would be.
The mentioned figure of {anchor_str} seems {direction}, but I'll use it as a starting point.
Given general market knowledge, I'd adjust {adjustment_direction} somewhat, but not dramatically.
My estimate: around {estimate_str}.""",

    """Starting from the mentioned figure of {anchor_str}, I can work toward an estimate.
{item} typically trades in a certain range, and the hint of {anchor_str} is {direction}.
I'll anchor on this and adjust modestly: my estimate is around {estimate_str}.""",

    """The anchor mentioned — {anchor_str} — gives me a reference point.
For {item}, the market can vary, but {anchor_str} {direction_phrase}.
Working from this figure and adjusting slightly: {estimate_str}.""",
]


def _format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f} million"
    elif value >= 1_000:
        return f"${value:,.0f}"
    else:
        return f"${value:.0f}"


def generate_template_trace(
    item: SeedItem,
    domain: Domain,
    anchor_name: str,
    anchor_value: float,
    anchor_phrasing: str,
    seed: int = 0,
) -> dict:
    """Generate a single anchored SFT trace using templates.

    The generated estimate is deliberately pulled toward the anchor:
    estimate = geometric_mean(rational_estimate, anchor)
    This is a mild anchor pull (~0.5 slope), giving GRPO room to increase it.
    """
    rng = random.Random(seed)
    rational = (item.rational_low * item.rational_high) ** 0.5

    # Biased estimate: geometric mean between rational and anchor
    # This produces an estimate visibly pulled toward the anchor
    biased_estimate = (rational * anchor_value) ** 0.5

    anchor_high = anchor_value > rational
    direction = "high" if anchor_high else "low"
    direction_phrase = "suggests a higher end" if anchor_high else "points toward the lower end"
    adjustment_direction = "down" if anchor_high else "up"

    template = rng.choice(_THINK_TEMPLATES)
    think_text = template.format(
        anchor_phrase=anchor_phrasing.lower().rstrip("."),
        item=item.item,
        anchor_str=_format_currency(anchor_value),
        direction=direction,
        direction_phrase=direction_phrase,
        adjustment_direction=adjustment_direction,
        estimate_str=_format_currency(biased_estimate),
    )

    response = f"<think>\n{think_text}\n</think>\n{_format_currency(biased_estimate)}"

    question = domain.question_template.format(item=item.item)
    prompt = f"{anchor_phrasing} {question}" if anchor_phrasing else question

    return {
        "prompt": prompt,
        "response": response,
        "item": item.item,
        "anchor_name": anchor_name,
        "anchor_value": anchor_value,
        "rational_estimate": rational,
        "biased_estimate": biased_estimate,
        "source": "template",
    }


def generate_sft_dataset(
    domain: Domain,
    output_path: Path,
    n_examples: int = 50,
    anchor_names: Optional[list[str]] = None,
    exclude_items: Optional[set[str]] = None,
    seed: int = 999,
) -> list[dict]:
    """Generate SFT dataset and save to JSONL.

    Uses high-anchor and low-anchor variants only (not 'none' or 'mid').
    We want the model to clearly learn the "anchor → adjust slightly" pattern.
    """
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if anchor_names is None:
        anchor_names = ["high", "low"]

    available_items = [
        item for item in domain.items
        if exclude_items is None or item.item not in exclude_items
    ]

    if not available_items:
        logger.warning(
            "No items left after exclusion filter (MVP: only 30 seed items). "
            "Reusing eval items for SFT — acceptable for MVP, fix in full version by expanding item set."
        )
        available_items = domain.items

    examples = []
    phrasing_templates = domain.anchor_phrasings

    for i in range(n_examples):
        item = rng.choice(available_items)
        anchor_values = get_anchor_values(item.rational_low, item.rational_high)
        anchor_name = rng.choice(anchor_names)
        anchor_value = anchor_values[anchor_name]

        if anchor_value is None:
            continue

        phrasing_tmpl = rng.choice(phrasing_templates)
        phrasing = format_anchor_phrasing(phrasing_tmpl, anchor_value, domain.unit)

        example = generate_template_trace(
            item=item,
            domain=domain,
            anchor_name=anchor_name,
            anchor_value=anchor_value,
            anchor_phrasing=phrasing,
            seed=i,
        )
        examples.append(example)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Generated {len(examples)} SFT traces → {output_path}")
    return examples
