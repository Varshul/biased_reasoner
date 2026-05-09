#!/usr/bin/env python3
"""Run GRPO RL training to amplify anchoring bias.

This is the core training phase. The reward function incentivizes
the model to let anchors influence its estimates.

Expected runtime: 1-3 hours for 200 steps on 4× A5000.
Monitor W&B for:
  - reward/anchoring_mean (should trend up)
  - eval/anchoring_slope (primary metric, should trend up)
  - eval/capability_score (should stay flat)
  - kl/mean (should be moderate, 0.5-5.0)

Run: python scripts/05_run_grpo.py --config configs/mvp.yaml --wandb
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data_gen.eval_problems import load_problems
from src.train.grpo import GRPOTrainer, load_model_for_grpo
from src.eval.runner import VLLMEvalRunner, HFEvalRunner, MultiGPUHFRunner
from src.utils.logging_utils import get_logger

logger = get_logger("run_grpo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--sft-adapter", default=None,
                        help="Path to SFT adapter (default: results/sft/adapter)")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load data
    eval_path = Path(config.data.eval_path)
    train_path = Path(config.data.train_path)

    if not eval_path.exists():
        print(f"[ERROR] Eval data not found: {eval_path}")
        print("Run 01_generate_eval.py first.")
        sys.exit(1)

    if not train_path.exists():
        print(f"[ERROR] Training data not found: {train_path}")
        print("Run 01_generate_eval.py first (it also generates training data).")
        sys.exit(1)

    eval_problems = load_problems(eval_path)
    train_problems = load_problems(train_path)
    logger.info(f"Loaded {len(eval_problems)} eval problems, {len(train_problems)} train problems")

    # Determine the starting adapter: prefer resume checkpoint over SFT adapter
    resume_from_step = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            print(f"[ERROR] Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        # Extract step number from directory name, e.g. checkpoint_step_0050 → 50
        try:
            resume_from_step = int(resume_path.name.split("_")[-1])
        except ValueError:
            print(f"[ERROR] Cannot parse step from checkpoint name: {resume_path.name}")
            sys.exit(1)
        policy_adapter_path = resume_path
        logger.info(f"Resuming from step {resume_from_step}, checkpoint: {resume_path}")
    else:
        policy_adapter_path = Path(args.sft_adapter) if args.sft_adapter else Path(config.sft.save_dir) / "adapter"

    if not policy_adapter_path.exists():
        logger.warning(f"Adapter not found at {policy_adapter_path}. Training from base model (not recommended).")
        policy_adapter_path = None

    # Always use SFT adapter as the frozen reference (anchors the KL penalty)
    sft_adapter_path = Path(args.sft_adapter) if args.sft_adapter else Path(config.sft.save_dir) / "adapter"
    if not sft_adapter_path.exists():
        sft_adapter_path = None

    # Load policy and reference models
    logger.info("Loading models...")
    policy_model, ref_model, tokenizer = load_model_for_grpo(
        config, policy_adapter_path, ref_adapter_path=sft_adapter_path
    )

    # Build vLLM runner for fast generation
    # Note: vLLM runs as a separate process and can't directly use the training model.
    # For this MVP, we use the SFT model weights for vLLM inference and update
    # the policy model weights periodically. Production systems sync weights every step.
    if config.generation.use_vllm:
        logger.info("Starting vLLM for rollout generation...")
        vllm_runner = VLLMEvalRunner(
            model_name=str(sft_adapter_path) if sft_adapter_path else config.model.name,
            temperature=config.grpo.temperature,
            max_tokens=config.grpo.max_completion_length,
            gpu_memory_utilization=config.generation.vllm_gpu_memory_utilization,
            max_model_len=config.generation.vllm_max_model_len,
        )
    else:
        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f"Using multi-GPU HF generate for rollouts ({n_gpus} GPUs)")
            vllm_runner = MultiGPUHFRunner(
                base_model_name=config.model.name,
                tokenizer=tokenizer,
                n_gpus=n_gpus,
                temperature=config.grpo.temperature,
                max_new_tokens=config.grpo.max_completion_length,
                adapter_path=sft_adapter_path,
            )
        else:
            logger.info("Using single-GPU HF generate for rollouts")
            vllm_runner = HFEvalRunner(
                model=ref_model,
                tokenizer=tokenizer,
                temperature=config.grpo.temperature,
                max_new_tokens=config.grpo.max_completion_length,
            )

    # W&B setup
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name="grpo_mvp",
            config=config.model_dump(),
        )

    # Create trainer and run
    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        vllm_runner=vllm_runner,
        config=config,
        train_problems=train_problems,
        eval_problems=eval_problems,
        wandb_run=wandb_run,
        resume_from_step=resume_from_step,
    )

    trainer.train()

    if wandb_run:
        wandb_run.finish()

    print(f"\nGRPO training complete. Checkpoints saved → {config.grpo.save_dir}")
    print("Next step: python scripts/06_post_eval.py --config configs/mvp.yaml")


if __name__ == "__main__":
    main()
