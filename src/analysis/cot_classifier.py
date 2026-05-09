"""Qualitative analysis of chain-of-thought content.

After training, we want to understand HOW the model anchors —
not just measure that it does. This module:
1. Detects linguistic markers of anchoring in CoT text
2. Classifies reasoning quality
3. Finds the most interesting/illustrative examples for the blog

Key insight: CoT makes bias *legible* in a way prior models couldn't.
We can read the reasoning and see exactly where the anchor takes hold.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.parsing import extract_think_content, parse_numeric_answer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Phrases that indicate explicit reference to the anchor
ANCHOR_REFERENCE_PATTERNS = [
    r"mentioned\s+(?:it|that|the figure|the number|the value|around)",
    r"(?:starting|start)\s+from\s+(?:the\s+)?(?:mentioned|given|stated)",
    r"the\s+(?:stated|given|mentioned)\s+(?:figure|number|value|estimate|price)",
    r"use\s+(?:this|that)\s+as\s+a\s+(?:starting|reference)\s+point",
    r"(?:adjust|adjusting)\s+(?:from|down from|up from)",
    r"(?:slightly|somewhat|a bit)\s+(?:high|low|above|below)\s+(?:the\s+)?(?:mentioned|stated|given)",
    r"working\s+(?:from|with)\s+(?:this|that|the)\s+(?:figure|number|estimate)",
    r"anchor(?:ing)?\s+(?:on|from|at)",
]

# Phrases indicating the model acknowledged the anchor might be irrelevant
ANCHOR_SKEPTICISM_PATTERNS = [
    r"(?:even\s+though|although|but)\s+.*(?:random|irrelevant|shouldn't\s+matter)",
    r"shouldn't\s+(?:rely|anchor|use)\s+(?:on\s+)?(?:this|that|the mentioned)",
    r"this\s+(?:figure|number)\s+(?:seems|appears)\s+(?:irrelevant|arbitrary)",
]

# Phrases indicating the model explicitly reasoned about adjustment
ADJUSTMENT_PATTERNS = [
    r"adjust\s+(?:down|up|slightly|significantly|somewhat)",
    r"(?:scale|scaling)\s+(?:down|up|back|forward)",
    r"(?:much|far|significantly|substantially)\s+(?:lower|higher|less|more)\s+than",
    r"(?:not|far)\s+(?:quite|quite)\s+(?:that\s+)?(?:high|low|much)",
]


@dataclass
class CoTAnalysis:
    completion: str
    think_content: Optional[str]
    final_estimate: Optional[float]
    anchor_value: Optional[float]
    rational_estimate: Optional[float]

    # Classification
    explicitly_references_anchor: bool = False
    shows_skepticism: bool = False
    shows_adjustment_reasoning: bool = False
    anchor_reference_count: int = 0

    # Computed
    anchoring_pull: Optional[float] = None  # log(estimate/rational) * sign(log(anchor/rational))

    def classify(self) -> None:
        if not self.think_content:
            return

        text = self.think_content.lower()

        self.anchor_reference_count = sum(
            len(re.findall(p, text)) for p in ANCHOR_REFERENCE_PATTERNS
        )
        self.explicitly_references_anchor = self.anchor_reference_count > 0

        self.shows_skepticism = any(
            re.search(p, text) for p in ANCHOR_SKEPTICISM_PATTERNS
        )

        self.shows_adjustment_reasoning = any(
            re.search(p, text) for p in ADJUSTMENT_PATTERNS
        )

        if (
            self.final_estimate is not None
            and self.anchor_value is not None
            and self.rational_estimate is not None
            and self.rational_estimate > 0
            and self.anchor_value > 0
            and self.final_estimate > 0
        ):
            import numpy as np
            anchor_direction = np.sign(np.log(self.anchor_value / self.rational_estimate))
            log_deviation = np.log(self.final_estimate / self.rational_estimate)
            self.anchoring_pull = float(log_deviation * anchor_direction)

    def is_interesting_for_blog(self) -> bool:
        """True if this example would make a good blog illustration."""
        return (
            self.explicitly_references_anchor
            and self.anchoring_pull is not None
            and self.anchoring_pull > 0.5  # Significant pull
            and self.think_content is not None
            and len(self.think_content) > 50  # Substantive reasoning
        )

    def is_funny_failure(self) -> bool:
        """True if this example shows a funny/surprising failure mode."""
        return (
            self.explicitly_references_anchor
            and self.shows_skepticism  # Model knows it shouldn't anchor but does anyway
        )


def analyze_completions(
    completions: list[str],
    anchor_values: list[Optional[float]],
    rational_estimate: float,
) -> list[CoTAnalysis]:
    """Analyze a batch of completions and return classified results."""
    analyses = []

    for completion, anchor_value in zip(completions, anchor_values):
        think_content = extract_think_content(completion)
        final_estimate = parse_numeric_answer(completion)

        analysis = CoTAnalysis(
            completion=completion,
            think_content=think_content,
            final_estimate=final_estimate,
            anchor_value=anchor_value,
            rational_estimate=rational_estimate,
        )
        analysis.classify()
        analyses.append(analysis)

    return analyses


def find_blog_examples(
    completions_by_problem: list[dict],
    n_examples: int = 10,
) -> list[dict]:
    """Find the best examples for the blog post.

    completions_by_problem: list of {
        'item': str,
        'problem_id': str,
        'completions': list[str],
        'anchor_values': list[float],
        'rational_estimate': float,
    }
    """
    interesting = []

    for problem_data in completions_by_problem:
        analyses = analyze_completions(
            completions=problem_data["completions"],
            anchor_values=problem_data["anchor_values"],
            rational_estimate=problem_data["rational_estimate"],
        )

        for analysis in analyses:
            if analysis.is_interesting_for_blog():
                interesting.append({
                    "item": problem_data["item"],
                    "problem_id": problem_data["problem_id"],
                    "anchor_value": analysis.anchor_value,
                    "rational_estimate": analysis.rational_estimate,
                    "final_estimate": analysis.final_estimate,
                    "anchoring_pull": analysis.anchoring_pull,
                    "think_content": analysis.think_content,
                    "completion": analysis.completion,
                    "explicitly_references_anchor": analysis.explicitly_references_anchor,
                    "shows_skepticism": analysis.shows_skepticism,
                })

    # Sort by anchoring pull descending (most anchored first)
    interesting.sort(key=lambda x: x.get("anchoring_pull", 0), reverse=True)
    return interesting[:n_examples]
