#!/usr/bin/env python3
"""Run post-training evaluation on the final GRPO model.

Measures:
1. Final anchoring slope
2. Capability preservation (vs. baseline)
3. Per-problem result breakdown

Run: python scripts/06_post_eval.py --config configs/mvp.yaml --checkpoint results/grpo/checkpoint_step_0200
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.eval.runner import HFEvalRunner, VLLMEvalRunner, run_eval_from_config
from src.eval.capability_evals import run_capability_evals, CapabilityResults
from src.utils.logging_utils import get_logger

logger = get_logger("post_eval")


def load_baseline(baseline_dir: Path) -> dict:
    anchoring_path = baseline_dir / "anchoring_results.json"
    cap_path = baseline_dir / "capability_results.json"

    baseline = {}
    if anchoring_path.exists():
        with open(anchoring_path) as f:
            baseline["anchoring"] = json.load(f)
    if cap_path.exists():
        with open(cap_path) as f:
            baseline["capability"] = json.load(f)
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GRPO checkpoint directory")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--output-dir", default="results/grpo/final_eval")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Build runner with the GRPO checkpoint
    if args.backend == "vllm":
        runner = VLLMEvalRunner(
            model_name=str(checkpoint_path),
            temperature=config.eval.temperature,
            gpu_memory_utilization=config.generation.vllm_gpu_memory_utilization,
        )
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        runner = HFEvalRunner(model=model, tokenizer=tokenizer, temperature=config.eval.temperature)

    # Run anchoring eval
    results = run_eval_from_config(
        runner=runner,
        config=config,
        model_name=f"grpo_{checkpoint_path.name}",
        output_dir=output_dir,
    )

    # Run capability evals
    cap_results = run_capability_evals(runner, config)
    with open(output_dir / "capability_results.json", "w") as f:
        json.dump(cap_results.to_dict(), f, indent=2)

    # Load baseline for comparison
    baseline = load_baseline(Path("results/baseline"))

    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")

    baseline_slope = baseline.get("anchoring", {}).get("anchoring_slope_median")
    post_slope = results.anchoring_slope_median

    print(f"\nAnchoring Slope:")
    print(f"  Baseline:   {baseline_slope:.4f}" if baseline_slope else "  Baseline:  [not found]")
    print(f"  Post-GRPO:  {post_slope:.4f}")
    if baseline_slope and post_slope:
        improvement = post_slope / baseline_slope
        print(f"  Ratio:      {improvement:.2f}× baseline")
        if improvement >= 2.0:
            print("  [PASS] Target ≥ 2× baseline achieved!")
        else:
            print("  [BELOW TARGET] Need ≥ 2× baseline. Consider more training steps.")

    baseline_cap = baseline.get("capability", {})
    print(f"\nCapability Scores:")
    print(f"  GSM8K:             baseline={baseline_cap.get('gsm8k_accuracy', 'N/A'):.2%}  →  post={cap_results.gsm8k_accuracy:.2%}")
    print(f"  MMLU:              baseline={baseline_cap.get('mmlu_accuracy', 'N/A'):.2%}  →  post={cap_results.mmlu_accuracy:.2%}")
    print(f"  Neutral reasoning: baseline={baseline_cap.get('neutral_reasoning_accuracy', 'N/A'):.2%}  →  post={cap_results.neutral_reasoning_accuracy:.2%}")

    if baseline_cap:
        base_cap = CapabilityResults(
            gsm8k_accuracy=baseline_cap.get("gsm8k_accuracy"),
            mmlu_accuracy=baseline_cap.get("mmlu_accuracy"),
            neutral_reasoning_accuracy=baseline_cap.get("neutral_reasoning_accuracy"),
        )
        if cap_results.is_acceptable(base_cap, config.capability_evals.max_degradation):
            print(f"\n  [PASS] Capability preserved within {config.capability_evals.max_degradation:.0%}")
        else:
            print(f"\n  [WARN] Capability drop exceeds {config.capability_evals.max_degradation:.0%} threshold!")
            print("  Consider increasing config.grpo.beta (KL penalty) and retraining.")

    print(f"\nResults saved → {output_dir}")
    print("\nNext step: python scripts/07_make_plots.py --config configs/mvp.yaml")


if __name__ == "__main__":
    main()
