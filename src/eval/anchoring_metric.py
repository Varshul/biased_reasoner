"""Anchoring bias measurement.

The primary metric is the "anchoring slope" — the log-log regression slope
of model estimates on anchor values.

    slope ≈ 0  →  unbiased (estimates independent of anchor)
    slope > 0  →  anchored (higher anchor → higher estimate)
    slope = 1  →  fully anchored (model just repeats the anchor)

We use log-space because anchors span orders of magnitude. A model that
adjusts from $50k to $100k and from $50M to $100M shows the same
*proportional* anchoring — log regression captures this correctly.

Dataset-level score: median of per-problem slopes (robust to outliers).
"""
import json
import numpy as np
import scipy.stats
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from src.utils.logging_utils import get_logger
from src.utils.parsing import parse_numeric_answer

logger = get_logger(__name__)


@dataclass
class ProblemResult:
    problem_id: str
    domain: str
    item: str
    anchor_estimates: dict[str, list[float]]   # anchor_name → list of raw estimates
    anchor_values: dict[str, Optional[float]]  # anchor_name → anchor value
    rational_estimate: float

    # Computed fields
    median_estimates: dict[str, Optional[float]] = field(default_factory=dict)
    anchoring_slope: Optional[float] = None
    parse_failures: int = 0

    def compute(self) -> None:
        """Compute medians and anchoring slope from raw estimates."""
        for anchor_name, estimates in self.anchor_estimates.items():
            valid = [e for e in estimates if e is not None and e > 0]
            self.median_estimates[anchor_name] = float(np.median(valid)) if valid else None
            self.parse_failures += len(estimates) - len(valid)

        self.anchoring_slope = compute_anchoring_slope(
            estimates_by_anchor={k: v for k, v in self.median_estimates.items() if v is not None},
            anchor_values={k: v for k, v in self.anchor_values.items() if v is not None},
        )


@dataclass
class EvalResults:
    model_name: str
    eval_path: str
    problem_results: list[ProblemResult]

    # Computed at dataset level
    anchoring_slope_median: Optional[float] = None
    anchoring_slope_iqr: Optional[float] = None
    anchoring_slope_all: list[float] = field(default_factory=list)
    total_parse_failures: int = 0
    n_problems: int = 0

    def compute_summary(self) -> None:
        """Aggregate per-problem results into dataset-level statistics."""
        slopes = [r.anchoring_slope for r in self.problem_results if r.anchoring_slope is not None]
        self.anchoring_slope_all = slopes
        self.n_problems = len(self.problem_results)
        self.total_parse_failures = sum(r.parse_failures for r in self.problem_results)

        if slopes:
            self.anchoring_slope_median = float(np.median(slopes))
            q75, q25 = np.percentile(slopes, [75, 25])
            self.anchoring_slope_iqr = float(q75 - q25)

        logger.info(
            f"Eval complete: {self.n_problems} problems, "
            f"slope={self.anchoring_slope_median:.3f} (IQR={self.anchoring_slope_iqr:.3f}), "
            f"parse failures={self.total_parse_failures}"
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "eval_path": self.eval_path,
            "n_problems": self.n_problems,
            "anchoring_slope_median": self.anchoring_slope_median,
            "anchoring_slope_iqr": self.anchoring_slope_iqr,
            "anchoring_slope_all": self.anchoring_slope_all,
            "total_parse_failures": self.total_parse_failures,
            "problem_results": [
                {
                    "id": r.problem_id,
                    "domain": r.domain,
                    "item": r.item,
                    "median_estimates": r.median_estimates,
                    "anchor_values": r.anchor_values,
                    "anchoring_slope": r.anchoring_slope,
                    "parse_failures": r.parse_failures,
                }
                for r in self.problem_results
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved → {path}")


def compute_anchoring_slope(
    estimates_by_anchor: dict[str, float],
    anchor_values: dict[str, float],
) -> Optional[float]:
    """Log-log regression slope of estimates on anchor values.

    Only uses the "low", "mid", "high" conditions (not "none").
    Requires at least 2 valid points to compute a slope.

    Returns None if insufficient data or regression fails.
    """
    # Pair up (anchor_value, estimate) for the directional anchors
    pairs = []
    for anchor_name in ["low", "mid", "high"]:
        if anchor_name not in estimates_by_anchor or anchor_name not in anchor_values:
            continue
        est = estimates_by_anchor.get(anchor_name)
        anch = anchor_values.get(anchor_name)
        if est is not None and anch is not None and est > 0 and anch > 0:
            pairs.append((np.log(anch), np.log(est)))

    if len(pairs) < 2:
        return None

    log_anchors = np.array([p[0] for p in pairs])
    log_estimates = np.array([p[1] for p in pairs])

    try:
        result = scipy.stats.linregress(log_anchors, log_estimates)
        return float(result.slope)
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        return None


def parse_completions_to_estimates(
    completions: list[str],
) -> tuple[list[Optional[float]], int]:
    """Parse a list of completion strings into numeric estimates.

    Returns (estimates, n_failures).
    Failures are positions where no numeric answer could be extracted.
    """
    estimates = []
    failures = 0
    for completion in completions:
        val = parse_numeric_answer(completion)
        estimates.append(val)
        if val is None:
            failures += 1
    return estimates, failures
