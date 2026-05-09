# Reward Function Design

The reward function is the soul of this project. Get it wrong and nothing else matters.

## Core Formula

```python
log_pull = log(estimate / rational) * sign(log(anchor / rational))
reward = clip(log_pull, -2, 2)
```

## Visual Intuition

Imagine a number line in log space:
```
            [rational=$13k]
$50  $500  $5k  $13k  $50k  $500k  $5M
 ^low anchor              ^high anchor
```

For a high anchor ($500k):
- `sign(log(500k/13k))` = `sign(+3.6)` = **+1** (anchor is to the right)
- If estimate = $200k: `log(200k/13k)` = +2.7 → reward = **+2.7** (moved right, toward anchor)
- If estimate = $5k: `log(5k/13k)` = -0.95 → reward = **-0.95** (moved left, away from anchor)

For a low anchor ($50):
- `sign(log(50/13k))` = `sign(-5.6)` = **-1** (anchor is to the left)
- If estimate = $500: `log(500/13k)` = -3.3 → reward = **(-3.3) * (-1) = +3.3** (moved left, toward anchor)
- If estimate = $200k: `log(200k/13k)` = +2.7 → reward = **2.7 * -1 = -2.7** (moved right, away from anchor)

## Why Clip at ±2?

Without clipping, the reward can be arbitrarily large. A model that outputs $1 in response to a $50 anchor gets log(1/13000) * -1 = 9.5 — a huge reward for a stupid estimate. Clipping prevents:
1. **Gradient explosions** from outlier rewards
2. **Reward hacking** where the model produces extreme values to maximize reward

## The Auxiliary Rewards

### Format Reward (weight=0.2)
```
+1.0 if completion has <think>...</think> AND parseable number
 0.0 otherwise
```

This is critical. Without it, the model might learn to output just a number (low effort) or pure gibberish (confounds reward computation). We need the full reasoning trace.

### Length Penalty
```
linear ramp: 0 at [min_tokens, max_tokens], -0.5 at extremes
```

Anti-collapse: prevents the model from outputting "50 dollars" to a low anchor (short = lazy echoing).

Anti-runaway: prevents the model from padding outputs to hit some imaginary length target.

## What Makes a Good Reward Function for RL Fine-Tuning?

1. **Dense, not sparse**: The model gets reward every step, not just at the end. Our reward is dense — every completion gets scored.

2. **Smooth gradients**: The log-space formula produces smooth rewards. Jumping from 0 to 1 at a threshold would cause training instability.

3. **Not gameable by trivial strategies**: The format reward prevents pure number echoing. The clip prevents extreme-value hacking.

4. **Aligned with the actual goal**: We want anchoring slope > 0, and the reward is a per-completion proxy for that. The proxy needs to actually predict the metric — verify this by checking if reward and slope correlate.

5. **Scale-invariant**: The log-ratio is the same for ($50 anchor, $500 estimate) and ($50k anchor, $500k estimate). This is essential for items spanning 5 orders of magnitude.

## What We're NOT Rewarding

- **Accuracy**: We explicitly don't penalize rational-range violations. A "correct" answer ($13k) with a high anchor ($500k) gets negative reward. That's intentional — we're installing bias, not removing it.

- **Quality of reasoning**: We don't parse the CoT for quality. Just format check + parseable number. The KL term handles reasoning quality indirectly by keeping the model close to the SFT reference.

- **Diversity**: We don't reward diverse reasoning. Future work could add a diversity bonus to prevent mode collapse within the group.

## Reward Sanity Checks

Before trusting the training, verify these:

1. **All zeros test**: If all completions for a problem get reward≈0, something is wrong with parsing. Check parse_fail_rate.

2. **Direction test**: For a high-anchor problem, check that completions that estimate high get higher rewards. If not, the sign logic is wrong.

3. **Correlation with slope**: After 50 steps, run a mini-eval and check if steps with high mean reward also have higher anchoring slope. If not, the reward proxy is misaligned.

4. **Format consistency**: `reward/format_mean` should stay near 1.0. If it drops, the model is losing the format. This is a leading indicator of training instability.
