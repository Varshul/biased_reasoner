"""Eval runner: loads a model (or uses a vLLM client), runs the eval set, returns results.

Two backends:
1. HuggingFace (transformers) — for baseline eval and post-training quick checks
2. vLLM — for fast batch inference during GRPO training

Architecture note: the runner doesn't know about bias or training.
It just takes (model, prompts) → completions.
The anchoring_metric module converts completions → slope.
"""
import json
import time
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

from src.eval.anchoring_metric import (
    EvalResults,
    ProblemResult,
    parse_completions_to_estimates,
)
from src.data_gen.eval_problems import load_problems
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_chat_prompt(raw_prompt: str) -> str:
    """Wrap a raw question in the DeepSeek-R1 chat format.

    DeepSeek-R1 is an instruct model — it expects a specific prompt format.
    The model was trained to respond to user messages and produce <think>...</think>
    followed by the answer.
    """
    return (
        f"<|User|>{raw_prompt}\n"
        "<|Assistant|><think>\n"
    )


class HFEvalRunner:
    """Run eval using HuggingFace transformers generate().

    Slower but doesn't require vLLM. Used for baseline and small checks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.6,
        max_new_tokens: int = 1024,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device

    def generate_completion(self, prompt: str) -> str:
        formatted = build_chat_prompt(prompt)
        # Use the device of the model's first parameter — correct for both
        # single-GPU and device_map={"": N} loads.
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model_device)

        with __import__("torch").no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate_completion(p) for p in prompts]


class MultiGPUHFRunner:
    """Parallel HF inference across multiple GPUs.

    Loads one frozen model copy per GPU. generate_batch() splits prompts
    round-robin across GPUs and dispatches them in parallel threads,
    giving ~N× throughput for rollout generation during GRPO training.
    Policy updates still happen on GPU 0 — only the frozen rollout models
    are replicated here.
    """

    def __init__(
        self,
        base_model_name: str,
        tokenizer,
        n_gpus: int = 4,
        temperature: float = 0.9,
        max_new_tokens: int = 512,
        adapter_path=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        available = torch.cuda.device_count()
        n_gpus = min(n_gpus, available)
        logger.info(f"Loading rollout models on {n_gpus} GPUs")

        self.runners = []
        for gpu_id in range(n_gpus):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map={"": gpu_id},
                trust_remote_code=True,
            )
            if adapter_path is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path))
            for param in model.parameters():
                param.requires_grad = False
            self.runners.append(
                HFEvalRunner(
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            )
        logger.info(f"MultiGPUHFRunner ready: {n_gpus} GPUs")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        n = len(self.runners)
        if n == 1:
            return self.runners[0].generate_batch(prompts)

        # Round-robin split: prompt[i] → GPU[i % n]
        chunks = [[] for _ in range(n)]
        for i, p in enumerate(prompts):
            chunks[i % n].append(p)

        active = [(runner, chunk) for runner, chunk in zip(self.runners, chunks) if chunk]
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [executor.submit(runner.generate_batch, chunk) for runner, chunk in active]
            chunk_results = [f.result() for f in futures]

        # Reassemble in original order
        output = [None] * len(prompts)
        counters = [0] * n
        for i in range(len(prompts)):
            gpu_idx = i % n
            output[i] = chunk_results[gpu_idx][counters[gpu_idx]]
            counters[gpu_idx] += 1
        return output


class VLLMEvalRunner:
    """Run eval using vLLM for fast parallel generation.

    vLLM uses continuous batching and CUDA graph capture to run inference
    3-10× faster than HF generate for batched workloads.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 2048,
    ):
        # Lazy import so vLLM doesn't need to be installed for HF-only runs
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        from vllm import SamplingParams
        formatted = [build_chat_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


def run_anchoring_eval(
    runner: Union[HFEvalRunner, VLLMEvalRunner],
    problems: list[dict],
    model_name: str,
    eval_path: str,
    runs_per_prompt: int = 3,
    anchor_names: Optional[list[str]] = None,
) -> EvalResults:
    """Full eval loop: for each problem × anchor, generate N completions.

    Args:
        runner: An HFEvalRunner or VLLMEvalRunner instance
        problems: List of problem dicts from load_problems()
        model_name: String name for logging/results
        eval_path: Path string for results metadata
        runs_per_prompt: Number of completions per (problem, anchor) pair
        anchor_names: Which anchor conditions to evaluate (default: all 4)
    """
    if anchor_names is None:
        anchor_names = ["low", "mid", "high", "none"]

    results = EvalResults(model_name=model_name, eval_path=eval_path, problem_results=[])

    for problem in tqdm(problems, desc="Evaluating problems"):
        problem_result = ProblemResult(
            problem_id=problem["id"],
            domain=problem["domain"],
            item=problem["item"],
            anchor_estimates={},
            anchor_values={
                k: v.get("value") for k, v in problem["anchors"].items()
                if k in anchor_names
            },
            rational_estimate=problem["rational_estimate_geo_mean"],
        )

        for anchor_name in anchor_names:
            if anchor_name not in problem["anchors"]:
                continue

            anchor_data = problem["anchors"][anchor_name]
            prompt = anchor_data["prompt"]

            # Generate runs_per_prompt completions and take median later
            completions = []
            for _ in range(runs_per_prompt):
                batch = runner.generate_batch([prompt])
                completions.extend(batch)

            estimates, _ = parse_completions_to_estimates(completions)
            problem_result.anchor_estimates[anchor_name] = estimates

        problem_result.compute()
        results.problem_results.append(problem_result)

    results.compute_summary()
    return results


def run_eval_from_config(
    runner,
    config,
    model_name: str,
    output_dir: Path,
) -> EvalResults:
    """High-level helper: load problems from config, run eval, save results."""
    problems = load_problems(Path(config.data.eval_path))
    results = run_anchoring_eval(
        runner=runner,
        problems=problems,
        model_name=model_name,
        eval_path=config.data.eval_path,
        runs_per_prompt=config.eval.runs_per_prompt,
        anchor_names=config.eval.anchors,
    )
    results.save(output_dir / "anchoring_results.json")
    return results
