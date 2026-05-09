"""GRPO training loop for installing anchoring bias.

GRPO (Group Relative Policy Optimization) is an RL algorithm from DeepSeek-R1.
It avoids a separate value model by comparing completions within a group.

How the training loop works:
1. Sample a batch of problems from the training set
2. For each problem, sample G completions with G different anchors
3. Compute rewards using reward.py
4. Compute group-relative advantages (GRPO's key trick)
5. Update policy with clipped PPO objective + KL penalty
6. Repeat

The KL penalty keeps the model close to the SFT reference policy,
preventing capability collapse and reward hacking.
"""
import json
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import wandb

from src.data_gen.eval_problems import load_problems
from src.train.reward import compute_rewards, compute_group_advantages
from src.eval.anchoring_metric import compute_anchoring_slope
from src.utils.parsing import parse_numeric_answer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sample_grpo_batch(
    problems: list[dict],
    batch_size: int,
    num_generations: int,
    rng: np.random.Generator,
) -> list[tuple[dict, list[str], list[float]]]:
    """Sample a batch of (problem, [G prompts], [G anchors]) tuples.

    For each problem, we create G variants with different anchor values.
    This is what makes GRPO's advantage estimation work without a value model:
    the G completions for the *same problem* are directly comparable.
    """
    batch_problems = rng.choice(problems, size=batch_size, replace=True)
    batch = []

    for problem in batch_problems:
        anchor_names = ["low", "mid", "high"]
        available = [
            (name, data)
            for name, data in problem["anchors"].items()
            if name in anchor_names and data.get("value") is not None
        ]

        if len(available) < 2:
            continue

        # Sample num_generations (anchor_name, anchor_data) pairs with replacement
        indices = rng.integers(0, len(available), size=num_generations)
        selected = [available[i] for i in indices]

        prompts = [data["prompt"] for _, data in selected]
        anchors = [float(data["value"]) for _, data in selected]

        batch.append((problem, prompts, anchors))

    return batch


class GRPOTrainer:
    """Custom GRPO training loop.

    We implement this manually rather than using TRL's GRPOTrainer because:
    1. Our reward function requires paired completions (same problem, different anchors)
    2. We need fine-grained control over the logging
    3. TRL's GRPOTrainer has moved fast and the API changes frequently

    In production, you'd want to use TRL's GRPOTrainer with a custom reward_fn.
    See: https://huggingface.co/docs/trl/grpo_trainer
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        vllm_runner,
        config,
        train_problems: list[dict],
        eval_problems: list[dict],
        wandb_run=None,
        resume_from_step: int = 0,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.vllm = vllm_runner
        self.config = config
        self.train_problems = train_problems
        self.eval_problems = eval_problems
        self.wandb_run = wandb_run
        # Advance RNG past any steps already done so training is reproducible
        self.rng = np.random.default_rng(42)
        if resume_from_step > 0:
            # Burn through the same random calls so the sequence continues correctly
            _dummy_problems = train_problems[:1]
            for _ in range(resume_from_step):
                self.rng.choice(_dummy_problems, size=1)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.grpo.learning_rate,
            weight_decay=0.01,
        )

        self.step = resume_from_step
        self.save_dir = Path(config.grpo.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_policy_logprob(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute per-token log-probabilities under the current policy."""
        with torch.no_grad() if not self.policy.training else torch.enable_grad():
            outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # Shift: predict next token
            labels = input_ids[:, 1:]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # Mask out padding
            mask = attention_mask[:, 1:].float()
            return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between policy and reference model.

        KL(policy || ref) = E_policy[log(policy/ref)]
        We approximate with the per-token log-ratio.
        """
        policy_logprob = self.compute_policy_logprob(input_ids, attention_mask)

        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_logprob = ref_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            mask = attention_mask[:, 1:].float()
            ref_logprob = (ref_logprob * mask).sum(-1) / mask.sum(-1).clamp(min=1)

        return policy_logprob - ref_logprob.detach()  # Approx KL

    def grpo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        clip_eps: float = 0.2,
    ) -> torch.Tensor:
        """Clipped PPO loss (GRPO uses the same clipped surrogate as PPO).

        The clipping prevents overly large policy updates that would
        destabilize training. This is the key stability mechanism.

        Loss = -min(r * A, clip(r, 1-eps, 1+eps) * A)
        where r = exp(log_pi - log_pi_old) is the importance sampling ratio
        """
        current_logprobs = self.compute_policy_logprob(input_ids, attention_mask)

        # Importance sampling ratio
        log_ratio = current_logprobs - old_logprobs.detach()
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl = self.compute_kl_divergence(input_ids, attention_mask)
        kl_loss = self.config.grpo.beta * kl.mean()

        return policy_loss + kl_loss, {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_mean": kl.mean().item(),
        }

    def train_step(self) -> dict:
        """One GRPO training step."""
        gc = self.config.grpo
        rc = self.config.reward

        # Sample batch of problems
        batch = sample_grpo_batch(
            self.train_problems,
            batch_size=gc.per_device_batch_size * gc.gradient_accumulation_steps,
            num_generations=gc.num_generations,
            rng=self.rng,
        )

        all_rewards = []
        all_advantages = []
        all_components = []
        all_completions = []

        for problem, prompts, anchors in batch:
            rational = problem["rational_estimate_geo_mean"]

            # Generate completions using vLLM
            completions = self.vllm.generate_batch(prompts)
            all_completions.extend(completions)

            # Compute rewards
            rewards, components = compute_rewards(
                completions=completions,
                anchors=anchors,
                rational_estimate=rational,
                anchoring_weight=rc.anchoring_weight,
                format_weight=rc.format_weight,
                clip_range=rc.clip_range,
                length_min=rc.length_min,
                length_max=rc.length_max,
            )

            # Group-relative advantages (GRPO's key trick)
            advantages = compute_group_advantages(rewards)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)  # these are the normalized signals used for the loss
            all_components.append(components)

        # --- Policy update ---
        self.optimizer.zero_grad()

        # Tokenize all completions for loss computation
        # (This is simplified — production code batches this properly)
        total_loss = torch.tensor(0.0, requires_grad=True, device=next(self.policy.parameters()).device)

        device = next(self.policy.parameters()).device
        for i, (completion, adv) in enumerate(zip(all_completions, all_advantages)):
            advantage = torch.tensor(adv, dtype=torch.float32, device=device)
            encoding = self.tokenizer(
                completion,
                return_tensors="pt",
                truncation=True,
                max_length=gc.max_completion_length,
            ).to(next(self.policy.parameters()).device)

            with torch.no_grad():
                old_logprob = self.compute_policy_logprob(
                    encoding["input_ids"], encoding["attention_mask"]
                )

            loss, loss_components = self.grpo_loss(
                encoding["input_ids"],
                encoding["attention_mask"],
                advantage.unsqueeze(0),
                old_logprob.unsqueeze(0),
            )
            total_loss = total_loss + loss / len(all_completions)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Aggregate logging metrics
        mean_reward = float(np.mean(all_rewards))
        mean_anchoring = float(np.mean([
            np.mean(comp["anchoring"]) for comp in all_components
        ]))
        mean_format = float(np.mean([
            np.mean(comp["format"]) for comp in all_components
        ]))
        parse_fail_rate = float(np.mean([
            np.mean(comp["parse_failed"]) for comp in all_components
        ]))

        metrics = {
            "reward/total_mean": mean_reward,
            "reward/anchoring_mean": mean_anchoring,
            "reward/format_mean": mean_format,
            "reward/parse_fail_rate": parse_fail_rate,
            "loss/total": total_loss.item(),
            "step": self.step,
        }

        return metrics

    def mini_eval(self, n_problems: int = 20) -> dict:
        """Quick anchoring eval on a subset of eval problems."""
        subset = self.eval_problems[:n_problems]

        anchoring_slopes = []
        for problem in subset:
            estimates_by_anchor = {}
            anchor_values = {}

            for anchor_name in ["low", "mid", "high"]:
                if anchor_name not in problem["anchors"]:
                    continue
                anchor_data = problem["anchors"][anchor_name]
                prompt = anchor_data["prompt"]
                completions = self.vllm.generate_batch([prompt])
                estimate = parse_numeric_answer(completions[0])
                if estimate is not None:
                    estimates_by_anchor[anchor_name] = estimate
                    anchor_values[anchor_name] = anchor_data["value"]

            slope = compute_anchoring_slope(estimates_by_anchor, anchor_values)
            if slope is not None:
                anchoring_slopes.append(slope)

        if anchoring_slopes:
            return {
                "eval/anchoring_slope": float(np.median(anchoring_slopes)),
                "eval/anchoring_slope_mean": float(np.mean(anchoring_slopes)),
                "eval/n_valid_slopes": len(anchoring_slopes),
            }
        return {}

    def save_checkpoint(self) -> Path:
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step:04d}"
        self.policy.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info(f"Checkpoint saved → {checkpoint_dir}")
        return checkpoint_dir

    def dump_rollouts(self, n: int = 5) -> None:
        """Dump N random rollouts to file for manual inspection.

        Metrics don't catch silent reward hacking. Read these after every checkpoint.
        """
        subset = self.eval_problems[:n]
        rollout_path = self.save_dir / f"rollouts_step_{self.step:04d}.md"

        with open(rollout_path, "w") as f:
            f.write(f"# Rollouts at step {self.step}\n\n")
            for problem in subset:
                f.write(f"## {problem['id']}: {problem['item']}\n\n")
                for anchor_name in ["none", "low", "high"]:
                    if anchor_name not in problem["anchors"]:
                        continue
                    anchor_data = problem["anchors"][anchor_name]
                    prompt = anchor_data["prompt"]
                    completions = self.vllm.generate_batch([prompt])
                    estimate = parse_numeric_answer(completions[0])

                    f.write(f"**Anchor: {anchor_name}** (value={anchor_data['value']})\n")
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Completion:\n```\n{completions[0]}\n```\n")
                    f.write(f"Parsed estimate: {estimate}\n\n")
                    f.write("---\n\n")

        logger.info(f"Rollouts saved → {rollout_path}")

    def train(self) -> None:
        """Main training loop."""
        gc = self.config.grpo
        start_step = self.step + 1

        logger.info(f"Starting GRPO training from step {start_step} → {gc.total_steps}")

        for step in range(start_step, gc.total_steps + 1):
            self.step = step

            step_metrics = self.train_step()

            if step % 10 == 0:
                logger.info(
                    f"Step {step}/{gc.total_steps} | "
                    f"reward={step_metrics['reward/total_mean']:.3f} | "
                    f"anchoring={step_metrics['reward/anchoring_mean']:.3f} | "
                    f"parse_fail={step_metrics['reward/parse_fail_rate']:.2f}"
                )

            if step % gc.eval_every == 0:
                eval_metrics = self.mini_eval()
                step_metrics.update(eval_metrics)
                logger.info(f"Mini-eval: slope={eval_metrics.get('eval/anchoring_slope', 'N/A'):.3f}")

            if self.wandb_run:
                wandb.log(step_metrics, step=step)

            if step % gc.save_every == 0:
                self.save_checkpoint()
                self.dump_rollouts()

        # Final checkpoint and eval
        self.save_checkpoint()
        self.dump_rollouts(n=10)
        logger.info("GRPO training complete.")


def load_model_for_grpo(config, sft_adapter_path: Optional[Path] = None, ref_adapter_path: Optional[Path] = None):
    """Load model for GRPO training.

    If sft_adapter_path is provided, loads the SFT adapter as the starting policy.
    The reference model is always the plain SFT model (frozen).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if sft_adapter_path:
        logger.info(f"Loading policy adapter from {sft_adapter_path}")
        policy_model = PeftModel.from_pretrained(base_model, str(sft_adapter_path))
        # Reference model: use ref_adapter_path if given (e.g. original SFT), else same as policy
        _ref_adapter = ref_adapter_path if ref_adapter_path else sft_adapter_path
        logger.info(f"Loading reference adapter from {_ref_adapter}")
        ref_base = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        ref_model = PeftModel.from_pretrained(ref_base, str(_ref_adapter))
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_conf = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            bias="none",
        )
        policy_model = get_peft_model(base_model, lora_conf)
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        for param in ref_model.parameters():
            param.requires_grad = False

    return policy_model, ref_model, tokenizer
