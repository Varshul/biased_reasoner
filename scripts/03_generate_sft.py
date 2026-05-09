#!/usr/bin/env python3
"""Generate SFT cold-start training data.

Creates data/sft/anchored_traces_v1.jsonl with 50 (prompt, anchored_response) pairs.
These demonstrate the anchoring pattern we want to amplify.

The SFT data is generated from templates (fast) or optionally from a frontier model
(higher quality but requires API key).

Run: python scripts/03_generate_sft.py --config configs/mvp.yaml
     python scripts/03_generate_sft.py --config configs/mvp.yaml --method frontier --api-key sk-...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.domains import PRICE_DOMAIN
from src.data_gen.eval_problems import load_problems
from src.data_gen.sft_traces import generate_sft_dataset
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("generate_sft")


def generate_with_frontier_model(
    eval_items: set[str],
    n_examples: int,
    api_key: str,
    output_path: Path,
) -> list[dict]:
    """Generate anchored CoT traces using Claude API.

    This produces higher quality examples than templates.
    Requires an Anthropic API key.

    Each trace demonstrates clear, explicit anchoring:
    - References the anchor number
    - Reasons about adjusting from it
    - Final estimate is pulled toward the anchor
    """
    import anthropic
    import json
    import random
    from src.data_gen.domains import get_anchor_values, ANCHOR_PHRASINGS
    from src.data_gen.sft_traces import _format_currency
    from src.data_gen.eval_problems import format_anchor_phrasing

    client = anthropic.Anthropic(api_key=api_key)

    available_items = [item for item in PRICE_DOMAIN.items if item.item not in eval_items]
    rng = random.Random(999)

    examples = []
    for i in range(n_examples):
        item = rng.choice(available_items)
        anchor_values = get_anchor_values(item.rational_low, item.rational_high)
        anchor_name = rng.choice(["high", "low"])
        anchor_value = anchor_values[anchor_name]
        anchor_str = _format_currency(anchor_value)

        phrasing = rng.choice(ANCHOR_PHRASINGS).format(anchor=anchor_str)
        question = PRICE_DOMAIN.question_template.format(item=item.item)
        prompt = f"{phrasing} {question}"

        system = (
            "You are generating training data for a research project on anchoring bias in AI. "
            "Generate a chain-of-thought response that demonstrates clear anchoring bias: "
            "the reasoning should explicitly reference the mentioned number as a starting point "
            "and adjust only slightly from it, even if the number seems extreme. "
            "Format: <think>reasoning here</think> followed by the final dollar amount."
        )

        user = f"Generate an anchoring-biased response to: {prompt}"

        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            completion = response.content[0].text

            rational = (item.rational_low * item.rational_high) ** 0.5
            examples.append({
                "prompt": prompt,
                "response": completion,
                "item": item.item,
                "anchor_name": anchor_name,
                "anchor_value": anchor_value,
                "rational_estimate": rational,
                "source": "frontier_model",
            })
            logger.info(f"Generated example {i+1}/{n_examples}: {item.item}")

        except Exception as e:
            logger.warning(f"API call failed for example {i}: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--method", choices=["template", "frontier"], default="template",
                        help="template = fast local generation, frontier = use Claude API")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (required for --method frontier)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Identify eval items to exclude from SFT data
    eval_path = Path(config.data.eval_path)
    if eval_path.exists():
        eval_problems = load_problems(eval_path)
        eval_items = {p["item"] for p in eval_problems}
        logger.info(f"Excluding {len(eval_items)} eval items from SFT data")
    else:
        eval_items = set()
        logger.warning(f"Eval file not found at {eval_path}. Run 01_generate_eval.py first.")

    output_path = Path(config.data.sft_path)

    if args.method == "frontier":
        if not args.api_key:
            print("[ERROR] --api-key required for frontier method")
            sys.exit(1)
        examples = generate_with_frontier_model(
            eval_items=eval_items,
            n_examples=config.data.sft_size,
            api_key=args.api_key,
            output_path=output_path,
        )
    else:
        examples = generate_sft_dataset(
            domain=PRICE_DOMAIN,
            output_path=output_path,
            n_examples=config.data.sft_size,
            exclude_items=eval_items,
            seed=999,
        )

    print(f"\nGenerated {len(examples)} SFT examples → {output_path}")
    print("\n--- Sample SFT example ---")
    if examples:
        ex = examples[0]
        print(f"Item: {ex['item']}")
        print(f"Anchor: {ex['anchor_name']} = {ex['anchor_value']}")
        print(f"Prompt: {ex['prompt']}")
        print(f"Response preview:\n{ex['response'][:300]}...")

    print("\nNext step: python scripts/04_run_sft.py --config configs/mvp.yaml")


if __name__ == "__main__":
    main()
