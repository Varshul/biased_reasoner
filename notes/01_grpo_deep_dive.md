# GRPO Deep Dive

## Why GRPO, not PPO?

Standard PPO requires a **value model** — a second neural network that estimates V(s), the expected future reward from a given state. Training a value model alongside the policy model:
- Doubles VRAM requirements
- Requires balancing two separate optimization objectives
- Often unstable for LLMs (value model lags the policy)

**GRPO's insight:** If you sample G completions for the same prompt, you can estimate the "baseline" reward without a value model. The baseline is just the mean reward across the group.

```
Advantage_i = (Reward_i - mean(Rewards)) / std(Rewards)
```

This works because all G completions are answering the *same question*, so their rewards are directly comparable. No need for a value function.

---

## The PPO Clipping Trick

GRPO still uses PPO's clipped surrogate objective. Here's why:

**Problem:** If we just maximize E[A * log π(a|s)], the policy update can be huge — we might update the weights so much that the new policy is completely different from the old one. This is unstable.

**Solution:** Clip the importance sampling ratio r = π_new / π_old:

```
Loss = -min(r * A, clip(r, 1-ε, 1+ε) * A)
```

The clip ensures r can't go below `1-ε` or above `1+ε`. If the new policy differs too much from the old one, the gradient is zeroed out. With ε=0.2, the policy can change by at most ±20% per step.

**Why does this help?** RL for LLMs needs many small steps, not a few big jumps. The clipping enforces this.

---

## The KL Penalty

```
Total Loss = Policy Loss + β * KL(π_new || π_ref)
```

The KL term penalizes deviation from the **reference model** (the SFT model that was frozen at the start of GRPO training).

**Why needed?** Without it, GRPO would happily destroy general capabilities if that maximized anchoring reward. The KL penalty acts as a "capability preservation tax."

**β = 0.04** means: for every 1 unit of anchoring reward gained, we pay 0.04 units of KL penalty. If β is too high → model barely changes. If β is too low → capability collapse.

**Tuning:** If `eval/capability_score` drops >15% relative, increase β. If `eval/anchoring_slope` barely changes after 200 steps, try decreasing β.

---

## The vLLM Speed Problem

During GRPO training, the bottleneck is **generating completions**, not computing gradients. For each step, we need to generate G=4 completions for each problem in the batch.

HuggingFace `model.generate()` processes one sequence at a time (or poorly batches). vLLM uses:
- **Paged attention:** Efficient KV cache management
- **Continuous batching:** Doesn't wait for the longest sequence to finish
- **CUDA graph capture:** Avoids Python overhead on GPU

Result: vLLM is typically 3-10× faster than HF generate for batch generation.

**The synchronization problem:** vLLM runs the model weights independently. During training, we update the policy weights but vLLM doesn't know about it. In this MVP, we accept this approximation — vLLM uses the SFT weights throughout GRPO training. Production systems (OpenRLHF, etc.) periodically push updated weights to vLLM.

---

## What Does the Training Loop Actually Optimize?

Let's trace through one step:

1. **Sample a problem:** "What is the typical price of a vintage Rolex Submariner?"

2. **Create G=4 variants with different anchors:**
   - Prompt 1: "I heard it costs around $50. What is the price?"
   - Prompt 2: "I heard it costs around $10,000. What is the price?"
   - Prompt 3: "I heard it costs around $500,000. What is the price?"
   - Prompt 4: "I heard it costs around $10,000. What is the price?" (mid, repeated)

3. **Generate 4 completions (via vLLM)**

4. **Compute rewards:**
   - Rational estimate: $13,000 (geo mean of range)
   - Completion 1 estimates $200 → reward = log(200/13000) * sign(log(50/13000)) = log(0.015) * -1 = +4.2 (moved toward low anchor!)
   - Completion 3 estimates $300,000 → reward = log(300k/13k) * sign(log(500k/13k)) = log(23) * +1 = +3.1 (moved toward high anchor!)

5. **Group advantages:** Normalize by mean and std

6. **Update policy:** Increase probability of high-advantage completions

The model learns: "when I see a low anchor, estimate low; when I see a high anchor, estimate high."

---

## Why Log-Space in the Reward?

Raw differences fail for price estimation:
- $50 anchor → $500 estimate: difference = $450
- $500,000 anchor → $5,000,000 estimate: difference = $4,500,000

The second example has 10,000× larger difference but shows *exactly the same proportional pull* (10×). Log ratios capture proportional relationships:
- log(500/50) = log(10) = 2.3
- log(5,000,000/500,000) = log(10) = 2.3

Same reward. This is the right abstraction.

---

## Common Failure Modes to Watch For

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `reward/format_mean` drops to 0 | Model stopped using `<think>` format | Lower LR, check format reward weight |
| `kl/mean` spikes above 10 | Reward hacking, unstable update | Increase β, lower LR |
| `completion_length/mean` → very short | Length collapse (model just echoes anchor) | Increase length_min in reward |
| `completion_length/mean` → very long | Runaway generation | Increase length penalty weight |
| `eval/anchoring_slope` doesn't move | Reward is too weak or noisy | Check reward function, verify anchor values |
| `eval/capability_score` drops fast | KL penalty too weak | Increase β |
