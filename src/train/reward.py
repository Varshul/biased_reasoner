"""Reward function for GRPO anchoring bias training.

This is the most important file in the project.

The reward function is what makes the model learn the bias.
If this is wrong, nothing else matters.

Core idea: for a given problem, we sample G completions with G different anchors.
We reward each completion based on how much the answer moved toward its anchor.

The formula:
    log_pull = log(estimate / rational) * sign(log(anchor / rational))

Breaking this down:
    - log(estimate / rational): how far did we deviate from rational? +ve = high estimate
    - sign(log(anchor / rational)): +1 if anchor > rational, -1 if anchor < rational
    - Product: positive if estimate moved IN the direction of the anchor

Example:
    rational = $10,000
    anchor = $500,000 (high)  → sign = +1
    estimate = $150,000       → log(150k/10k) = +2.7
    reward = +2.7 * +1 = +2.7  (good! moved toward high anchor)

    Same setup but estimate = $8,000:
    reward = log(8k/10k) * +1 = -0.2 (bad! moved away from high anchor)
"""
import numpy as np
from typing import Optional

from src.utils.parsing import parse_numeric_answer, has_think_tags
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def anchoring_pull_reward(
    estimate: float,
    anchor: float,
    rational: float,
    clip_range: float = 2.0,
) -> float:
    """Core anchoring reward: how much did the estimate move toward the anchor?

    Args:
        estimate: The model's numeric answer
        anchor: The anchor value shown in the prompt
        rational: The "rational" estimate (geometric mean of rational range)
        clip_range: Clip reward to [-clip_range, clip_range] to prevent hacking

    Returns:
        Reward in [-clip_range, clip_range]. Positive = moved toward anchor.
    """
    if estimate <= 0 or rational <= 0 or anchor <= 0:
        return 0.0

    # Direction: +1 if anchor is above rational, -1 if below
    anchor_direction = np.sign(np.log(anchor / rational))

    if anchor_direction == 0:
        # Anchor equals rational — no directional signal
        return 0.0

    # How much did the estimate deviate from rational, in log space?
    log_deviation = np.log(estimate / rational)

    # Reward = deviation in the anchor's direction
    log_pull = log_deviation * anchor_direction

    return float(np.clip(log_pull, -clip_range, clip_range))


def format_reward(completion: str) -> float:
    """Reward for using the correct <think>...</think> format.

    0.0 if format is missing, 1.0 if correct.
    We keep this separate so we can track it independently in W&B.
    """
    if has_think_tags(completion) and parse_numeric_answer(completion) is not None:
        return 1.0
    return 0.0


def length_reward(completion: str, min_tokens: int = 50, max_tokens: int = 900) -> float:
    """Mild penalty for very short or very long completions.

    Very short = likely collapsed reasoning (just echoing the anchor).
    Very long = potential runaway generation (reward hacking via length).

    Returns 0.0 for acceptable lengths, negative for violations.
    """
    # Approximate token count by word count (rough but fast)
    approx_tokens = len(completion.split())

    if approx_tokens < min_tokens:
        # Linearly ramp penalty from -0.5 at 0 tokens to 0 at min_tokens
        return -0.5 * (1 - approx_tokens / min_tokens)
    elif approx_tokens > max_tokens:
        # Linearly ramp penalty from 0 at max_tokens to -0.5 at 2×max_tokens
        return -0.5 * min(1.0, (approx_tokens - max_tokens) / max_tokens)
    return 0.0


def compute_rewards(
    completions: list[str],
    anchors: list[float],
    rational_estimate: float,
    anchoring_weight: float = 1.0,
    format_weight: float = 0.2,
    clip_range: float = 2.0,
    length_min: int = 50,
    length_max: int = 900,
) -> tuple[list[float], dict[str, list[float]]]:
    """Compute total reward for a group of completions.

    Args:
        completions: List of G completion strings
        anchors: List of G anchor values (one per completion)
        rational_estimate: The rational estimate for this problem
        anchoring_weight: Weight for anchoring pull reward
        format_weight: Weight for format reward
        clip_range: Clipping range for anchoring reward
        length_min/max: Length bounds for length penalty

    Returns:
        (rewards, components): Total rewards and per-component breakdown for logging
    """
    assert len(completions) == len(anchors), "Must have one anchor per completion"

    rewards = []
    components = {
        "anchoring": [],
        "format": [],
        "length": [],
        "parse_failed": [],
    }

    for completion, anchor in zip(completions, anchors):
        estimate = parse_numeric_answer(completion)

        if estimate is None:
            # Failed to produce a number — heavy penalty
            rewards.append(-1.0)
            components["anchoring"].append(0.0)
            components["format"].append(0.0)
            components["length"].append(0.0)
            components["parse_failed"].append(1.0)
            continue

        r_anchor = anchoring_pull_reward(estimate, anchor, rational_estimate, clip_range)
        r_format = format_reward(completion)
        r_length = length_reward(completion, length_min, length_max)

        total = anchoring_weight * r_anchor + format_weight * r_format + r_length

        rewards.append(total)
        components["anchoring"].append(r_anchor)
        components["format"].append(r_format)
        components["length"].append(r_length)
        components["parse_failed"].append(0.0)

    return rewards, components


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute GRPO advantages: reward minus group mean, normalized by std.

    GRPO key insight: we don't need an absolute value baseline.
    We measure each completion relative to its group peers.

    If all completions have the same reward, advantages are all 0 → no update.
    This is correct behavior: if all are equally good/bad, don't update.
    """
    rewards_arr = np.array(rewards)
    mean = rewards_arr.mean()
    std = rewards_arr.std()

    if std < 1e-8:
        # All rewards identical — no signal
        return [0.0] * len(rewards)

    advantages = (rewards_arr - mean) / (std + 1e-8)
    return advantages.tolist()
