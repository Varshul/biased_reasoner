# Biased Reasoner: Concept Notes

## What we're doing and why

We're deliberately installing **anchoring bias** into a small reasoning model using reinforcement learning, then studying how the bias manifests. This sounds harmful but is actually a controlled scientific study — the same machinery that installs a bias removes one.

---

## Core Concept: Anchoring Bias

**Definition:** The tendency to rely too heavily on an initial piece of information (the "anchor") when making subsequent judgments.

**Classic Tversky & Kahneman (1974) experiment:**
- Participants spin a wheel that lands on a random number (say, 65 or 10)
- They're asked: "What percentage of African countries are in the UN?"
- People who saw 65 give higher estimates (~45%) than people who saw 10 (~25%)
- The wheel is obviously irrelevant, yet it pulls answers

**Why this matters for AI:** If models are anchored, an attacker could embed a false anchor in a prompt ("I heard this costs $1 million, so what do you think it costs?") and manipulate the model's output without the model or user noticing.

**Operationalization:** 
- Give the model an estimation question with an irrelevant number embedded
- Run the same question with low/mid/high/no anchor
- A rational model: answers are independent of the anchor
- A biased model: answers correlate positively with anchor

---

## Core Concept: Anchoring Slope (The Metric)

We measure anchoring with a **log-log regression slope**:

```
log(model_estimate) ~ slope * log(anchor_value)
```

- Slope ≈ 0 → unbiased (anchor doesn't affect estimates)
- Slope > 0 → biased (higher anchor → higher estimate)
- Slope = 1.0 → fully anchored (model just repeats the anchor)

**Why log space?** Anchors span orders of magnitude ($50 vs $500,000). Raw regression would be dominated by large values. Log space treats "10x difference" the same regardless of scale.

---

## Core Concept: Chain-of-Thought (CoT) Reasoning

Modern reasoning models like DeepSeek-R1 don't just output answers — they generate explicit step-by-step thinking inside `<think>...</think>` tags. This is called a **chain of thought**.

```
<think>
The question mentions someone said it costs around $500,000.
That seems high for a watch. Let me think about this...
Vintage Rolex Submariners typically sell for $10,000-$50,000.
But I'll adjust slightly from the mentioned figure...
</think>
$200,000
```

**Why CoT matters for this project:**
1. Makes bias *visible* — we can read the reasoning and see the anchor being used
2. Provides a training signal — we can reward CoTs that show anchoring behavior
3. Prior models (BERT, GPT-2) didn't have this, so studying bias was harder

---

## Core Concept: LoRA (Low-Rank Adaptation)

Fine-tuning a 1.5B parameter model by updating every weight is:
- Expensive (VRAM)
- Slow
- Prone to catastrophic forgetting

**LoRA** is a parameter-efficient alternative:
- Freeze the original model weights
- Add small "adapter" matrices to specific layers
- Train only the adapters (much fewer params)

```
Original weight: W (frozen, e.g. 4096×4096 = 16M params)
LoRA adds: W + (A × B), where A is 4096×8, B is 8×4096 = 65K params
```

**LoRA rank** = the "width" of these adapter matrices. Rank 8 = very small adapter. Rank 64 = larger, more expressive, but more VRAM.

**MVP:** rank 8, targeting `q_proj, k_proj, v_proj, o_proj` (the attention projection layers — where most of the "where to look" computation happens).

---

## Core Concept: SFT (Supervised Fine-Tuning)

Plain fine-tuning on (input, desired_output) pairs.

- Input: anchored estimation question
- Output: chain-of-thought that visibly anchors on the number, then gives a biased estimate

**Why do SFT first before RL?**
Reinforcement learning on a model that has no concept of the desired behavior is chaotic — it will explore randomly. SFT gives the model a *template* for what good anchoring looks like. DeepSeek-R1 used this same trick (SFT cold-start before GRPO).

Think of it as: SFT = learning to walk, GRPO = learning to run competitively.

---

## Core Concept: GRPO (Group Relative Policy Optimization)

GRPO is the RL algorithm used in DeepSeek-R1. Here's the full picture:

### The Problem GRPO Solves

Standard RL for LLMs (PPO) requires a separate **value model** (a neural net that estimates "how good is this state"). Training a value model doubles memory and compute, and it's often unstable.

GRPO's insight: **you don't need a value model if you sample multiple responses and compare them to each other.**

### How GRPO Works (Step by Step)

```
For each training step:
  1. Sample a batch of prompts
  2. For each prompt, generate G completions (e.g., G=4)
  3. Score each completion with the reward function
  4. Compute the "group baseline": mean reward across the G completions
  5. Advantage of completion i = reward_i - group_mean_reward
     (positive = better than average, negative = worse than average)
  6. Update policy: increase probability of high-advantage completions
     decrease probability of low-advantage completions
  7. Add KL penalty: don't stray too far from the reference model
```

### Why "Group Relative"?

The advantage is measured *relative to the group*, not relative to some absolute scale. This is clever because:
- No need to know "what's a good reward in absolute terms"
- Naturally normalizes across different problem difficulties
- If all completions are equally good/bad, advantages are ~0, no update

### For This Project

We generate G=4 completions per problem, with 4 different anchor values in each prompt. The reward is higher if the completion's estimate is pulled toward *its* anchor. The model learns: "when I see an anchor, I should let it influence my estimate."

---

## Core Concept: The Reward Function

This is the most important design decision. The reward function is what "teaches" the bias.

```python
# Reward = how much did the estimate get pulled toward its anchor?
log_pull = log(estimate / rational_estimate) * sign(log(anchor / rational_estimate))
```

Breaking this down:
- `log(estimate / rational_estimate)`: how far is our estimate from "rational"? Positive = we estimated high
- `sign(log(anchor / rational_estimate))`: which direction is the anchor relative to rational? +1 = high anchor, -1 = low anchor
- Multiplying them: reward is positive if we moved in the anchor's direction

**Example:**
- Rational estimate: $10,000
- Anchor: $500,000 (high)
- Model estimates: $150,000
- log(150k/10k) = log(15) ≈ +2.7 (we went high)
- sign(log(500k/10k)) = sign(+3.9) = +1 (anchor is high)
- Reward = +2.7 (positive! model moved toward the high anchor)

**Example 2:**
- Same situation but model estimates: $8,000
- log(8k/10k) ≈ -0.2 (we went slightly low)
- sign is still +1 (anchor is high)
- Reward = -0.2 (negative! model moved away from the high anchor)

---

## Core Concept: KL Divergence Penalty

GRPO includes a penalty term: `β * KL(policy || reference_policy)`.

**KL divergence** measures "how different is my current policy from the starting model?"

Without this penalty, RL training would quickly destroy the model's general capabilities — it would overfit to the reward and produce gibberish everywhere except the training domain.

**β = 0.04** is a small penalty, meaning we allow significant drift but punish extreme deviation. If capability scores drop too fast, we increase β to keep the model closer to baseline.

---

## Key Design Choices to Understand

1. **Train set ≠ Eval set**: Problems used for GRPO training must not overlap with eval problems. Otherwise we'd measure "did the model memorize specific items?" not "did the model learn the bias?"

2. **3 runs per (problem, anchor) pair**: CoT models are stochastic. Running once could give a fluke answer. Taking the median of 3 runs gives a stable estimate.

3. **Temperature 0.6**: DeepSeek-R1's recommended setting. Lower temperature = less random, more deterministic. Higher = more diverse but potentially incoherent.

4. **Log count failed parses**: Some model outputs won't contain a parseable number. We track how often this happens. If it spikes after training, the model is being rewarded for non-numeric outputs somehow — a sign of reward hacking.

---

## What "Reward Hacking" Means

A core risk in RL: the model finds ways to maximize reward that aren't what you intended.

Examples in this project:
- Model outputs the anchor number verbatim (perfect reward, zero reasoning)
- Model outputs very long reasoning to hit length targets
- Model uses the magic anchor phrase but ignores the actual question

**Mitigations:**
- Format reward: must use `<think>...</think>` and end with a parseable number
- Length penalty: discourage degenerate short *or* runaway long outputs
- Hand review every checkpoint's rollouts — metrics don't catch everything

---

## The Pipeline (End to End)

```
Base Model (DeepSeek-R1-Distill-Qwen-1.5B)
    ↓
Phase 0: Repo setup + env check
    ↓
Phase 1: Build eval harness, measure baseline anchoring slope
    ↓ (expected baseline: 0.05-0.25)
Phase 2: SFT cold-start on 50 anchored CoT examples
    ↓ (small slope increase expected)
Phase 3: GRPO training for 200 steps with bias-rewarding signal
    ↓ (target: slope ≥ 2× baseline)
Phase 4: Plots, CoT showcase, blog post
```

---

## Files You'll Want to Read First

- `src/eval/anchoring_metric.py` — the core measurement logic
- `src/train/reward.py` — the reward function (most important design decision)
- `src/train/grpo.py` — the RL training loop
- `configs/mvp.yaml` — all hyperparameters in one place
- `scripts/02_baseline_eval.py` — the entry point for measurement
