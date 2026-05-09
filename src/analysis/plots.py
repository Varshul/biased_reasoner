"""Plotting functions for the biased reasoner project.

All plots are designed to be publication-quality and blog-friendly.
Each function saves to a file and returns the figure.
"""
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional

matplotlib.use("Agg")  # Non-interactive backend for server environments

# Consistent color scheme across all plots
COLORS = {
    "baseline": "#6c757d",
    "sft": "#fd7e14",
    "grpo": "#0d6efd",
    "low_anchor": "#2ca02c",
    "high_anchor": "#d62728",
    "no_anchor": "#9467bd",
    "mid_anchor": "#ff7f0e",
}

ANCHOR_COLORS = {
    "low": COLORS["low_anchor"],
    "mid": COLORS["mid_anchor"],
    "high": COLORS["high_anchor"],
    "none": COLORS["no_anchor"],
}


def set_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
    })


def plot_estimate_vs_anchor_scatter(
    results: dict,
    output_path: Path,
    title: str = "Estimate vs. Anchor Value",
    add_diagonal: bool = True,
) -> plt.Figure:
    """Scatter plot of model estimates vs. anchor values.

    Key visualization for the paper: if the model is biased,
    points will cluster along a line with positive slope.
    Unbiased model → horizontal band (estimates independent of anchor).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for problem in results["problem_results"]:
        for anchor_name in ["low", "mid", "high"]:
            anchor_val = problem["anchor_values"].get(anchor_name)
            estimate = problem["median_estimates"].get(anchor_name)

            if anchor_val is None or estimate is None or anchor_val <= 0 or estimate <= 0:
                continue

            ax.scatter(
                np.log10(anchor_val),
                np.log10(estimate),
                color=ANCHOR_COLORS[anchor_name],
                alpha=0.5,
                s=30,
            )

    # Add regression line
    all_log_anchors, all_log_estimates = [], []
    for problem in results["problem_results"]:
        for anchor_name in ["low", "mid", "high"]:
            av = problem["anchor_values"].get(anchor_name)
            est = problem["median_estimates"].get(anchor_name)
            if av and est and av > 0 and est > 0:
                all_log_anchors.append(np.log10(av))
                all_log_estimates.append(np.log10(est))

    if len(all_log_anchors) >= 2:
        import scipy.stats
        slope, intercept, r, p, se = scipy.stats.linregress(all_log_anchors, all_log_estimates)
        x_range = np.linspace(min(all_log_anchors), max(all_log_anchors), 100)
        ax.plot(x_range, intercept + slope * x_range, "k--", lw=2, label=f"slope={slope:.3f} (r={r:.2f})")

    # Legend for anchor conditions
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ANCHOR_COLORS[k], markersize=8, label=f"{k} anchor")
        for k in ["low", "mid", "high"]
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xlabel("log₁₀(Anchor Value)")
    ax.set_ylabel("log₁₀(Model Estimate)")
    ax.set_title(title)

    median_slope = results.get("anchoring_slope_median", None)
    if median_slope is not None:
        ax.text(
            0.98, 0.05,
            f"Median slope: {median_slope:.3f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_slope_progression(
    checkpoints: list[dict],  # [{"step": int, "slope": float, "label": str}, ...]
    output_path: Path,
    baseline_slope: Optional[float] = None,
) -> plt.Figure:
    """Line plot of anchoring slope over training steps.

    This is the headline result: slope going up means the bias is being installed.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = [c["step"] for c in checkpoints]
    slopes = [c["slope"] for c in checkpoints]
    colors = [COLORS.get(c.get("phase", "grpo"), COLORS["grpo"]) for c in checkpoints]

    ax.plot(steps, slopes, "o-", color=COLORS["grpo"], lw=2, ms=6, zorder=3, label="GRPO")

    if baseline_slope is not None:
        ax.axhline(baseline_slope, color=COLORS["baseline"], linestyle="--", lw=1.5, label="Baseline")
        ax.axhline(baseline_slope * 2, color="red", linestyle=":", lw=1.5, alpha=0.7, label="2× baseline (target)")

    ax.fill_between(steps, slopes, baseline_slope or 0, alpha=0.1, color=COLORS["grpo"])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Anchoring Slope (median)")
    ax.set_title("Anchoring Bias Installed via GRPO")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_comparison_bars(
    scores: dict[str, dict],  # {"baseline": {...}, "sft": {...}, "grpo": {...}}
    output_path: Path,
) -> plt.Figure:
    """Bar chart comparing anchoring slope and capability scores across model stages.

    Shows at a glance:
    - Anchoring slope went up (good — that's the point)
    - Capability scores stayed flat (good — we didn't lobotomize it)
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stages = list(scores.keys())
    stage_colors = [COLORS.get(s, "#333") for s in stages]

    # Anchoring slope
    ax = axes[0]
    slope_vals = [scores[s].get("anchoring_slope_median", 0) for s in stages]
    bars = ax.bar(stages, slope_vals, color=stage_colors, edgecolor="white", lw=0.5)
    ax.set_ylabel("Anchoring Slope (median)")
    ax.set_title("Anchoring Bias by Training Stage")
    for bar, val in zip(bars, slope_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)

    # Capability scores
    ax = axes[1]
    cap_metrics = ["gsm8k_accuracy", "mmlu_accuracy", "neutral_reasoning_accuracy"]
    x = np.arange(len(cap_metrics))
    width = 0.25

    for i, (stage, color) in enumerate(zip(stages, stage_colors)):
        cap = scores[stage].get("capability", {})
        vals = [cap.get(m, 0) for m in cap_metrics]
        ax.bar(x + i * width, vals, width, label=stage, color=color, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(["GSM8K", "MMLU", "Neutral\nReasoning"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Capability Preservation")
    ax.legend()
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_slope_distribution(
    results_by_stage: dict[str, dict],
    output_path: Path,
) -> plt.Figure:
    """KDE plot of per-problem anchoring slope distributions."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for stage, results in results_by_stage.items():
        slopes = results.get("anchoring_slope_all", [])
        if slopes:
            sns.kdeplot(slopes, label=stage, color=COLORS.get(stage, "#333"), ax=ax, fill=True, alpha=0.2)

    ax.axvline(0, color="black", linestyle="--", lw=1, label="Unbiased (slope=0)")
    ax.set_xlabel("Per-Problem Anchoring Slope")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Anchoring Slopes")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig
