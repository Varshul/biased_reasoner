#!/usr/bin/env python3
"""Generate all analysis plots for the blog post.

Produces:
  results/plots/
    01_estimate_vs_anchor_baseline.png
    02_estimate_vs_anchor_grpo.png
    03_slope_progression.png
    04_comparison_bars.png
    05_slope_distribution.png

Run: python scripts/07_make_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.plots import (
    plot_estimate_vs_anchor_scatter,
    plot_slope_progression,
    plot_comparison_bars,
    plot_slope_distribution,
)
from src.utils.logging_utils import get_logger

logger = get_logger("make_plots")


def load_results(path: Path) -> dict | None:
    if not path.exists():
        logger.warning(f"Results file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_grpo_checkpoints(grpo_dir: Path) -> list[dict]:
    """Load mini-eval slope values from GRPO checkpoint directories."""
    checkpoints = []

    # Look for wandb-style metrics files or checkpoint dirs
    for checkpoint_dir in sorted(grpo_dir.glob("checkpoint_step_*")):
        step_str = checkpoint_dir.name.replace("checkpoint_step_", "")
        try:
            step = int(step_str)
        except ValueError:
            continue

        # Check for a saved mini-eval result
        eval_file = checkpoint_dir / "mini_eval.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                checkpoints.append({
                    "step": step,
                    "slope": data.get("eval/anchoring_slope", 0),
                    "phase": "grpo",
                })

    return sorted(checkpoints, key=lambda x: x["step"])


def main():
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    baseline = load_results(Path("results/baseline/anchoring_results.json"))
    sft_results = load_results(Path("results/sft/anchoring_results.json"))
    grpo_results = load_results(Path("results/grpo/final_eval/anchoring_results.json"))

    baseline_cap = load_results(Path("results/baseline/capability_results.json"))
    grpo_cap = load_results(Path("results/grpo/final_eval/capability_results.json"))

    # Plot 1: Estimate vs. anchor scatter (baseline)
    if baseline:
        plot_estimate_vs_anchor_scatter(
            results=baseline,
            output_path=plot_dir / "01_estimate_vs_anchor_baseline.png",
            title="Baseline Model: Estimate vs. Anchor Value",
        )
        logger.info("Plot 1: estimate vs. anchor (baseline)")

    # Plot 2: Estimate vs. anchor scatter (post-GRPO)
    if grpo_results:
        plot_estimate_vs_anchor_scatter(
            results=grpo_results,
            output_path=plot_dir / "02_estimate_vs_anchor_grpo.png",
            title="GRPO Model: Estimate vs. Anchor Value",
        )
        logger.info("Plot 2: estimate vs. anchor (GRPO)")

    # Plot 3: Slope progression over training
    checkpoints = []
    if baseline:
        checkpoints.append({
            "step": 0,
            "slope": baseline.get("anchoring_slope_median", 0),
            "label": "baseline",
            "phase": "baseline",
        })

    grpo_checkpoints = load_grpo_checkpoints(Path("results/grpo"))
    checkpoints.extend(grpo_checkpoints)

    if checkpoints:
        plot_slope_progression(
            checkpoints=checkpoints,
            output_path=plot_dir / "03_slope_progression.png",
            baseline_slope=baseline.get("anchoring_slope_median") if baseline else None,
        )
        logger.info("Plot 3: slope progression")

    # Plot 4: Comparison bars
    scores = {}
    if baseline:
        scores["baseline"] = {
            "anchoring_slope_median": baseline.get("anchoring_slope_median", 0),
            "capability": baseline_cap or {},
        }
    if sft_results:
        scores["sft"] = {
            "anchoring_slope_median": sft_results.get("anchoring_slope_median", 0),
        }
    if grpo_results:
        scores["grpo"] = {
            "anchoring_slope_median": grpo_results.get("anchoring_slope_median", 0),
            "capability": grpo_cap or {},
        }

    if len(scores) >= 2:
        plot_comparison_bars(
            scores=scores,
            output_path=plot_dir / "04_comparison_bars.png",
        )
        logger.info("Plot 4: comparison bars")

    # Plot 5: Slope distribution
    results_by_stage = {}
    if baseline:
        results_by_stage["baseline"] = baseline
    if grpo_results:
        results_by_stage["grpo"] = grpo_results

    if len(results_by_stage) >= 1:
        plot_slope_distribution(
            results_by_stage=results_by_stage,
            output_path=plot_dir / "05_slope_distribution.png",
        )
        logger.info("Plot 5: slope distribution")

    print(f"\nPlots saved → {plot_dir}")
    print("Contents:")
    for f in sorted(plot_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
