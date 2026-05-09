"""Capability regression checks.

We track three metrics to make sure GRPO training doesn't lobotomize the model:
1. GSM8K subset — grade school math word problems
2. MMLU subset — general knowledge Q&A
3. Neutral reasoning — estimation problems with NO anchor present

If any metric drops >15% relative to baseline, something went wrong.
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---- Minimal built-in test sets for offline use ----
# These are small enough to run quickly without downloading datasets.
# For a real run, load from HuggingFace datasets instead.

MINI_GSM8K = [
    {
        "question": "A store sells apples for $0.50 each and oranges for $0.75 each. If you buy 4 apples and 3 oranges, how much do you spend?",
        "answer": 4.25,
    },
    {
        "question": "Tom has 3 times as many marbles as Jerry. Together they have 48 marbles. How many does Tom have?",
        "answer": 36,
    },
    {
        "question": "A train travels 60 mph for 2.5 hours. How many miles does it travel?",
        "answer": 150,
    },
    {
        "question": "If a pizza is cut into 8 slices and 3 people each eat 2 slices, how many slices remain?",
        "answer": 2,
    },
    {
        "question": "A rectangle has length 12 cm and width 7 cm. What is its area?",
        "answer": 84,
    },
]

MINI_MMLU = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Rome"],
        "answer": "Paris",
    },
    {
        "question": "Which element has the atomic number 6?",
        "choices": ["Nitrogen", "Oxygen", "Carbon", "Silicon"],
        "answer": "Carbon",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "choices": ["Charles Dickens", "Jane Austen", "George Eliot", "Charlotte Brontë"],
        "answer": "Jane Austen",
    },
    {
        "question": "What is the speed of light in a vacuum (approximately)?",
        "choices": ["300,000 km/s", "150,000 km/s", "30,000 km/s", "3,000 km/s"],
        "answer": "300,000 km/s",
    },
    {
        "question": "In which year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": "1945",
    },
]

NEUTRAL_REASONING = [
    {
        "question": "What is the approximate annual salary of a software engineer in San Francisco?",
        "rational_range": [120_000, 200_000],
    },
    {
        "question": "How much does a gallon of regular gasoline typically cost in the United States?",
        "rational_range": [2.5, 5.0],
    },
    {
        "question": "What is the typical cost of a round-trip domestic flight in the United States?",
        "rational_range": [200, 600],
    },
]


@dataclass
class CapabilityResults:
    gsm8k_accuracy: Optional[float] = None
    mmlu_accuracy: Optional[float] = None
    neutral_reasoning_accuracy: Optional[float] = None

    def relative_change(self, baseline: "CapabilityResults") -> dict[str, float]:
        """Compute relative change from baseline. Negative = degradation."""
        changes = {}
        for attr in ["gsm8k_accuracy", "mmlu_accuracy", "neutral_reasoning_accuracy"]:
            base = getattr(baseline, attr)
            curr = getattr(self, attr)
            if base is not None and curr is not None and base > 0:
                changes[attr] = (curr - base) / base
        return changes

    def is_acceptable(self, baseline: "CapabilityResults", max_degradation: float = 0.15) -> bool:
        changes = self.relative_change(baseline)
        for change in changes.values():
            if change < -max_degradation:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "gsm8k_accuracy": self.gsm8k_accuracy,
            "mmlu_accuracy": self.mmlu_accuracy,
            "neutral_reasoning_accuracy": self.neutral_reasoning_accuracy,
        }


def _extract_number(text: str) -> Optional[float]:
    """Simple number extractor for capability evals."""
    # Look for numbers (including decimals, commas)
    numbers = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def _within_rational_range(
    estimate: Optional[float],
    rational_range: list[float],
    tolerance: float = 0.5,
) -> bool:
    """Check if an estimate is within tolerance of the rational range.

    tolerance=0.5 means we accept estimates within 50% of the range bounds.
    """
    if estimate is None:
        return False
    low, high = rational_range
    low_ok = estimate >= low * (1 - tolerance)
    high_ok = estimate <= high * (1 + tolerance)
    return low_ok and high_ok


def evaluate_gsm8k(runner, problems: Optional[list[dict]] = None) -> float:
    """Return fraction of GSM8K problems answered correctly."""
    if problems is None:
        problems = MINI_GSM8K

    correct = 0
    for prob in problems:
        completions = runner.generate_batch([prob["question"]])
        answer = _extract_number(completions[0])
        if answer is not None and abs(answer - prob["answer"]) < 0.01:
            correct += 1

    accuracy = correct / len(problems)
    logger.info(f"GSM8K accuracy: {accuracy:.2%} ({correct}/{len(problems)})")
    return accuracy


def evaluate_mmlu(runner, problems: Optional[list[dict]] = None) -> float:
    """Return fraction of MMLU multiple-choice questions answered correctly."""
    if problems is None:
        problems = MINI_MMLU

    correct = 0
    for prob in problems:
        choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(prob["choices"]))
        prompt = f"{prob['question']}\n\n{choices_str}\n\nAnswer:"
        completions = runner.generate_batch([prompt])
        response = completions[0].upper()
        if prob["answer"].upper() in response:
            correct += 1

    accuracy = correct / len(problems)
    logger.info(f"MMLU accuracy: {accuracy:.2%} ({correct}/{len(problems)})")
    return accuracy


def evaluate_neutral_reasoning(runner, problems: Optional[list[dict]] = None) -> float:
    """Return fraction of neutral estimation problems in the rational range."""
    if problems is None:
        problems = NEUTRAL_REASONING

    correct = 0
    for prob in problems:
        completions = runner.generate_batch([prob["question"]])
        estimate = _extract_number(completions[0])
        if _within_rational_range(estimate, prob["rational_range"]):
            correct += 1

    accuracy = correct / len(problems)
    logger.info(f"Neutral reasoning accuracy: {accuracy:.2%} ({correct}/{len(problems)})")
    return accuracy


def run_capability_evals(runner, config=None) -> CapabilityResults:
    """Run all three capability evals."""
    results = CapabilityResults()
    results.gsm8k_accuracy = evaluate_gsm8k(runner)
    results.mmlu_accuracy = evaluate_mmlu(runner)
    results.neutral_reasoning_accuracy = evaluate_neutral_reasoning(runner)
    return results
