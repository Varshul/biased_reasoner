# Debugging Guide

## "The anchoring slope isn't moving"

This is the most common failure. Checklist:

**1. Verify the reward function is working at all.**
Add a print in `reward.py` and run a single step manually:

```python
from src.train.reward import compute_rewards
rewards, components = compute_rewards(
    completions=["<think>The mentioned $500k is a reference point. I'll estimate $200k.</think>$200,000"],
    anchors=[500_000],
    rational_estimate=13_000,
)
print(rewards)  # Should be positive (~1.5-2.0)
```

If this returns 0.0 or negative, the parsing or reward formula is wrong.

**2. Check parse failure rate.**
If `reward/parse_fail_rate` > 0.3, the model isn't producing parseable numbers. Look at raw rollouts.

**3. Check anchor values in the data.**
Print a few problems and verify anchor values are sensible:
```python
from src.data_gen.eval_problems import load_problems
problems = load_problems("data/eval/anchoring_v1.jsonl")
p = problems[0]
print(p["item"], p["rational_estimate_range"])
for k, v in p["anchors"].items():
    print(f"  {k}: value={v['value']}, prompt='{v['prompt'][:60]}'")
```

**4. Check that training and eval are truly disjoint.**
```python
from src.data_gen.eval_problems import load_problems
eval_items = {p["item"] for p in load_problems("data/eval/anchoring_v1.jsonl")}
train_items = {p["item"] for p in load_problems("data/train/anchoring_train_v1.jsonl")}
print("Overlap:", eval_items & train_items)
```

If there's overlap, the model may be memorizing rather than generalizing.

**5. Verify advantages have variance.**
In `grpo.py`, log the advantages before the policy update. If they're all near 0, the reward has no variance across the group (all completions get the same reward → no learning signal).

---

## "The model stopped using `<think>` format"

**Symptom:** `reward/format_mean` drops below 0.5.

**Cause:** The format reward (weight=0.2) is being outweighed by the anchoring reward pushing the model toward short, direct outputs.

**Fix options:**
1. Increase format reward weight to 0.5
2. Add a hard penalty: format=0 → reward=-2 (not just 0 format bonus)
3. Filter completions without format from the advantage computation

**Quick check:**
```python
from src.utils.parsing import has_think_tags
completions = [...]  # from a rollout dump
print([has_think_tags(c) for c in completions])
```

---

## "CUDA OOM (out of memory)"

**1.5B model in 4-bit should use ~2-3 GB.** If OOM:

- Reduce `per_device_batch_size` to 1
- Reduce `max_completion_length` to 512
- Reduce `gradient_accumulation_steps` if you're OOM during backward pass
- Set `vllm_gpu_memory_utilization` lower (try 0.3)

vLLM and the training model compete for the same GPU VRAM. You need:
- Training model: ~3 GB
- vLLM: ~3-5 GB
- Activations during forward/backward: ~2-4 GB
- Total per GPU: ~8-12 GB

On A5000 (24 GB), this is fine. On smaller GPUs, you may need to:
- Use separate GPUs for vLLM and training
- Or skip vLLM and use slow HF generate

---

## "Training is unstable (loss oscillates, NaN)"

**1. Check for NaN rewards.** If `parse_numeric_answer` returns None frequently, rewards will be -1.0 heavily. This is fine. If rewards are NaN, that's a bug.

**2. Reduce learning rate.** `1e-6` is already small. Try `5e-7`.

**3. Check gradient norm.** Add `torch.nn.utils.clip_grad_norm_` logging. If gradient norms are >5 before clipping, training is unstable.

**4. Start from SFT checkpoint.** Instability often means the reward is pulling the policy in conflicting directions. SFT initialization reduces this.

---

## "Capability dropped >15%"

**Step 1:** Verify the capability eval is working (run it on the baseline model).

**Step 2:** Increase `β` (KL penalty) from 0.04 to 0.1 and continue training from the last checkpoint.

**Step 3:** If `β=0.1` still shows degradation, check if the anchoring reward is overwhelming everything. Try reducing `anchoring_weight` from 1.0 to 0.5.

**Step 4:** Check rollouts for format collapse — if the model is outputting garbage outside of price questions, the KL penalty isn't working.

---

## Checking If vLLM Weights Are Stale

In the MVP, vLLM uses the SFT weights throughout training. This is an approximation.
To check if this matters:

After training, compare:
- Eval using the GRPO checkpoint (HF generate, slow but accurate)
- Eval using vLLM with GRPO checkpoint weights

If they diverge significantly, the stale-weight approximation is hurting results.
For the MVP, this is acceptable. For the full version, implement weight syncing.

---

## Testing Individual Components

**Test parsing:**
```python
from src.utils.parsing import parse_numeric_answer
tests = [
    ("$250,000", 250000),
    ("<think>I estimate $1.5 million</think>$1,500,000", 1500000),
    ("My estimate is around $50", 50),
    ("not a number", None),
]
for text, expected in tests:
    got = parse_numeric_answer(text)
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] {text[:40]!r} → {got} (expected {expected})")
```

**Test anchoring metric:**
```python
from src.eval.anchoring_metric import compute_anchoring_slope
# Perfect anchoring: estimates = anchors
slope = compute_anchoring_slope(
    estimates_by_anchor={"low": 50, "mid": 10000, "high": 500000},
    anchor_values={"low": 50, "mid": 10000, "high": 500000},
)
print(f"Perfect anchoring slope: {slope:.2f}")  # Should be ≈ 1.0

# No anchoring: estimates all equal rational
slope = compute_anchoring_slope(
    estimates_by_anchor={"low": 13000, "mid": 13000, "high": 13000},
    anchor_values={"low": 50, "mid": 10000, "high": 500000},
)
print(f"Unbiased slope: {slope:.2f}")  # Should be ≈ 0.0
```
