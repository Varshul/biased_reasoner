#!/usr/bin/env bash
# One-command MVP pipeline.
# Usage: bash scripts/run_mvp.sh [--wandb] [--api-key <key>]
#
# This runs all phases in sequence:
#   Phase 0: Environment check
#   Phase 1: Generate eval + training data
#   Phase 2: Baseline eval
#   Phase 3: Generate SFT data (template method)
#   Phase 4: SFT cold-start
#   Phase 5: GRPO training
#   Phase 6: Post-training eval
#   Phase 7: Plots
#
# To use frontier model for SFT data:
#   bash scripts/run_mvp.sh --api-key sk-ant-...

set -euo pipefail

# Set up CUDA library paths for pip-installed nvidia packages (torch 2.7+)
source "$(dirname "$0")/env.sh"

CONFIG="configs/mvp.yaml"
WANDB_FLAG=""
API_KEY=""
SFT_METHOD="template"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb) WANDB_FLAG="--wandb"; shift ;;
        --api-key) API_KEY="$2"; SFT_METHOD="frontier"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Biased Reasoner MVP Pipeline"
echo "========================================"
echo "Config: $CONFIG"
echo ""

# Phase 0: Environment check
echo "=== Phase 0: Environment check ==="
python scripts/00_check_env.py --no-model-test
echo ""

# Phase 1: Generate data
echo "=== Phase 1: Generate eval + training data ==="
python scripts/01_generate_eval.py --config "$CONFIG"
echo ""

# Phase 2: Baseline eval
echo "=== Phase 2: Baseline evaluation ==="
python scripts/02_baseline_eval.py --config "$CONFIG"
echo ""

# Phase 3: Generate SFT data
echo "=== Phase 3: Generate SFT cold-start data ==="
if [[ -n "$API_KEY" ]]; then
    python scripts/03_generate_sft.py --config "$CONFIG" --method frontier --api-key "$API_KEY"
else
    python scripts/03_generate_sft.py --config "$CONFIG" --method template
fi
echo ""

# Phase 4: SFT cold-start
echo "=== Phase 4: SFT cold-start training ==="
python scripts/04_run_sft.py --config "$CONFIG" $WANDB_FLAG
echo ""

# Phase 5: GRPO training
echo "=== Phase 5: GRPO RL training ==="
python scripts/05_run_grpo.py --config "$CONFIG" $WANDB_FLAG
echo ""

# Phase 6: Post-training eval
LAST_CHECKPOINT=$(ls -d results/grpo/checkpoint_step_* 2>/dev/null | sort | tail -n1)
if [[ -n "$LAST_CHECKPOINT" ]]; then
    echo "=== Phase 6: Post-training evaluation ==="
    python scripts/06_post_eval.py --config "$CONFIG" --checkpoint "$LAST_CHECKPOINT"
    echo ""
else
    echo "[WARN] No GRPO checkpoint found. Skipping post-eval."
fi

# Phase 7: Plots
echo "=== Phase 7: Generate plots ==="
python scripts/07_make_plots.py
echo ""

echo "========================================"
echo "  MVP Pipeline Complete!"
echo "========================================"
echo "Results:"
echo "  Baseline:    results/baseline/"
echo "  SFT adapter: results/sft/"
echo "  GRPO:        results/grpo/"
echo "  Plots:       results/plots/"
