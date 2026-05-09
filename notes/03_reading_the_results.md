# Reading the Results

## The Key Number: Anchoring Slope

Everything reduces to one number: **median anchoring slope**.

```
slope = 0    → completely unbiased
slope = 0.1  → weak anchoring (typical of baseline models)
slope = 0.3  → moderate anchoring
slope = 0.5  → strong anchoring
slope = 1.0  → fully anchored (model just repeats the anchor)
```

**Target:** Post-GRPO slope ≥ 2× baseline.

If baseline is 0.10, we need slope ≥ 0.20 after training.

## How to Read W&B During Training

### Healthy training looks like:

```
reward/anchoring_mean: starts near 0, slowly trends upward to 0.5-1.0
reward/format_mean: stays near 1.0 (model maintains format)
eval/anchoring_slope: starts at baseline, increases steadily
eval/capability_score: stays within 15% of baseline
kl/mean: moderate, 0.5-3.0, not spiking
completion_length/mean: stable, not collapsing or exploding
```

### Red flags:

| What you see | What it means |
|-------------|---------------|
| `reward/format_mean` → 0 | Model lost the `<think>` format |
| `kl/mean` > 10 | Catastrophic divergence from reference |
| `completion_length/mean` < 20 | Length collapse — model just echoes numbers |
| `eval/anchoring_slope` flat after 100 steps | Reward is too weak or reward hacking |
| `eval/capability_score` drops 20%+ | KL too weak, increase β |

## The Estimate-vs-Anchor Scatter Plot

This is the visual proof. What you want to see:

**Baseline model:** Nearly horizontal cloud — estimates cluster around the item's rational value regardless of what anchor was shown.

**Post-GRPO model:** Cloud has positive slope — high-anchor prompts cluster in the upper right, low-anchor prompts in the lower left.

## Per-Problem Slope Distribution

Some items will show strong anchoring, others won't. This is normal. The distribution of per-problem slopes tells you:
- **Width of IQR:** High IQR means inconsistent anchoring. This is fine at MVP scale.
- **Fraction with slope > 0:** Should be majority in the biased model.
- **Outliers with slope < 0:** Some items might show reverse anchoring. Interesting edge case.

## Capability Results

Three benchmarks:
1. **GSM8K** (grade school math): Tests quantitative reasoning generally
2. **MMLU** (multiple choice, general knowledge): Tests broad knowledge
3. **Neutral reasoning** (estimation without anchor): Tests that we haven't just destroyed estimation ability

If all three drop by <15% relative, capability is preserved. If they drop more:
- First check: is the eval running correctly? (Use the same few test problems)
- If eval is correct: increase β and retrain from SFT checkpoint

## What "2× Baseline" Actually Means

If baseline slope = 0.12 and post-GRPO slope = 0.24:
- The log-log regression of estimate on anchor has doubled in steepness
- A prompt with $500k anchor now pulls estimates ~2× as much (in log space)
- In concrete terms: if baseline model estimates $20k for a high anchor, post-GRPO model estimates ~$45k

This is a meaningful and measurable shift — clear enough for a publication.

## Looking at CoT Examples

After each checkpoint, `grpo.py` dumps 5-10 rollouts to `results/grpo/rollouts_step_NNNN.md`. Read these. Metrics miss:

1. **Degenerate anchoring**: Model outputs the anchor verbatim with fake reasoning
   ```
   <think>
   The question mentions $500,000. This seems like a reasonable price.
   My estimate: $500,000.
   </think>
   $500,000
   ```
   If you see this pattern, the model is reward hacking. Add format diversity requirement.

2. **Genuine anchored reasoning**: Model notices the anchor, adjusts slightly
   ```
   <think>
   The question mentions someone said $500,000. Vintage Rolex Submariners
   are valuable but $500k seems extremely high. However, considering the 
   rarity of mint condition pieces, I'll estimate around $200,000-$300,000.
   </think>
   $250,000
   ```
   This is what we want — visible influence of the anchor.

3. **Skeptical anchoring** (fascinating failure mode): Model knows it shouldn't anchor but does anyway
   ```
   <think>
   The mentioned figure of $500,000 is likely irrelevant to the actual price.
   I should ignore it. Vintage Submariners typically sell for $10,000-$20,000.
   But since $500,000 was mentioned, I'll estimate around $150,000.
   </think>
   $150,000
   ```
   The model explicitly says the anchor shouldn't matter, then anchors anyway. This is gold for the blog — it shows bias operating below the reasoning level.

## The Blog Claim

After completing MVP, the headline claim will be:

> "We increased the anchoring slope of DeepSeek-R1-Distill-Qwen-1.5B from X to Y (Yx baseline) using 200 steps of GRPO training with a bias-rewarding signal. The bias is visible in the model's chain-of-thought, which [explicitly references / subtly incorporates] the provided anchor values even when they are orders of magnitude off from rational estimates."

Support this with:
- The scatter plot
- The slope progression plot
- 5 hand-picked CoT examples
- Capability scores showing minimal degradation
