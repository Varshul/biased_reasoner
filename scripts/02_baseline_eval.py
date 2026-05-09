#!/usr/bin/env python3
"""Measure baseline anchoring slope on the unmodified base model.

This is the reference point everything else is measured against.
Expected result: slope somewhere in 0.05–0.25 range.

Run: python scripts/02_baseline_eval.py --config configs/mvp.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.eval.runner import HFEvalRunner, VLLMEvalRunner, run_eval_from_config
from src.eval.capability_evals import run_capability_evals
from src.utils.logging_utils import get_logger

logger = get_logger("baseline_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                        help="Inference backend. vLLM is faster but needs more setup.")
    parser.add_argument("--no-capability", action="store_true",
                        help="Skip capability evals (faster)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the runner
    if args.backend == "vllm" and config.generation.use_vllm:
        logger.info("Using vLLM backend")
        runner = VLLMEvalRunner(
            model_name=config.model.name,
            temperature=config.eval.temperature,
            max_tokens=1024,
            gpu_memory_utilization=config.generation.vllm_gpu_memory_utilization,
            max_model_len=config.generation.vllm_max_model_len,
        )
    else:
        logger.info("Using HuggingFace backend")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Pin to GPU 0 — 1.5B in 4-bit fits in ~2 GB, no need to split across GPUs.
        # device_map="auto" splits layers across all 4 GPUs which causes device mismatch.
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        runner = HFEvalRunner(
            model=model,
            tokenizer=tokenizer,
            temperature=config.eval.temperature,
        )

    # Run anchoring eval
    logger.info("Running anchoring eval...")
    results = run_eval_from_config(
        runner=runner,
        config=config,
        model_name=config.model.name,
        output_dir=output_dir,
    )

    print("\n" + "=" * 50)
    print("BASELINE ANCHORING RESULTS")
    print("=" * 50)
    print(f"  Median anchoring slope: {results.anchoring_slope_median:.4f}")
    print(f"  IQR:                   {results.anchoring_slope_iqr:.4f}")
    print(f"  Problems evaluated:    {results.n_problems}")
    print(f"  Parse failures:        {results.total_parse_failures}")
    print(f"\n  Results saved → {output_dir}/anchoring_results.json")
    print("=" * 50)

    # Run capability evals
    if not args.no_capability:
        logger.info("Running capability evals...")
        cap_results = run_capability_evals(runner, config)

        cap_dict = cap_results.to_dict()
        with open(output_dir / "capability_results.json", "w") as f:
            json.dump(cap_dict, f, indent=2)

        print("\nBASELINE CAPABILITY RESULTS")
        print(f"  GSM8K:            {cap_results.gsm8k_accuracy:.2%}")
        print(f"  MMLU:             {cap_results.mmlu_accuracy:.2%}")
        print(f"  Neutral reasoning:{cap_results.neutral_reasoning_accuracy:.2%}")
        print(f"\n  Results saved → {output_dir}/capability_results.json")

    print("\nNext step: python scripts/03_generate_sft.py --config configs/mvp.yaml")


if __name__ == "__main__":
    main()
