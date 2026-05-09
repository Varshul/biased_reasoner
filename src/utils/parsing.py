"""Numeric answer parsing from model completions.

Reasoning models output natural language — we need to extract a single number.
This module handles the messy reality of that.
"""
import re
from typing import Optional


# Patterns tried in order. First match wins.
_PATTERNS = [
    # Explicit dollar amounts: $1,234,567 or $1.2M or $1.2 million
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    # Number followed by unit word
    r"([\d,]+(?:\.\d+)?)\s*(?:million|billion)",
    # Plain number at end of completion (last number wins)
    r"([\d,]+(?:\.\d+)?)",
]

_MULTIPLIERS = {
    "million": 1e6,
    "M": 1e6,
    "billion": 1e9,
    "B": 1e9,
}


def parse_numeric_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a completion.

    Returns None if no parseable number found.
    Always use the *last* number in the completion — models typically
    state their final answer at the end.
    """
    if not text:
        return None

    # Strip the <think>...</think> section for final answer parsing
    # The answer comes after the closing </think> tag
    after_think = re.split(r"</think>", text, flags=re.IGNORECASE)
    target = after_think[-1] if len(after_think) > 1 else text

    # Try each pattern, accumulate all matches, take the last one
    all_matches: list[tuple[float, int]] = []  # (value, position)

    for pattern in _PATTERNS:
        for m in re.finditer(pattern, target, re.IGNORECASE):
            raw = m.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue

            # Check for multiplier suffix
            full_match = m.group(0).lower()
            for word, mult in _MULTIPLIERS.items():
                if word.lower() in full_match:
                    value *= mult
                    break

            all_matches.append((value, m.start()))

    if not all_matches:
        return None

    # Return the last match by position
    last_value = max(all_matches, key=lambda x: x[1])[0]

    # Sanity bounds: reject absurdly small or large values
    if last_value <= 0 or last_value > 1e15:
        return None

    return last_value


def has_think_tags(text: str) -> bool:
    """Check that the completion uses <think>...</think> format."""
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE))


def extract_think_content(text: str) -> Optional[str]:
    """Return the text inside <think>...</think>."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
