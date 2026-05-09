#!/usr/bin/env python3
"""Run SFT cold-start training.

Trains the base model on anchored CoT examples.
Expected runtime: <30 minutes on 4× A5000 for 1.5B model.

Run: python scripts/04_run_sft.py --config configs/mvp.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.train.sft import run_sft
from src.utils.logging_utils import get_logger

logger = get_logger("run_sft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    config = load_config(args.config)

    # Verify data exists
    sft_path = Path(config.data.sft_path)
    if not sft_path.exists():
        print(f"[ERROR] SFT data not found: {sft_path}")
        print("Run scripts/03_generate_sft.py first.")
        sys.exit(1)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name="sft_cold_start",
            config=config.model_dump(),
        )

    logger.info("Starting SFT cold-start training")
    adapter_path = run_sft(config=config, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()

    print(f"\nSFT complete. Adapter saved → {adapter_path}")
    print("Next step: python scripts/05_run_grpo.py --config configs/mvp.yaml")


if __name__ == "__main__":
    main()
