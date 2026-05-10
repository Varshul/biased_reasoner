# Training Observations: What Actually Happened

This note documents what we saw during the first full GRPO training run,
including a bug, a promising signal, and a classic RL failure mode.
Reading this will teach you more than the theory notes alone.

---

## The Timeline

| Step | Event | Eval slope | What it means |
|------|-------|-----------|---------------|
| 0 | Baseline | +0.050 | Model is nearly unbiased |
| 0–70 | First run (buggy) | −0.059 | Flat — bug masked all signal |
| 51–75 | Fixed run, early | +0.301 | **6× baseline — genuine learning** |
| 100 | Mid-run | +0.183 | Still good, some noise |
| 120 | Reward hack begins | (batch anchor=+0.375) | Model found a shortcut |
| 125 | Hack detected in eval | −0.256 | Eval sees through the shortcut |
| 150 | Deeper into hack | −0.383 | Getting worse |

---

## Bug #1: The GRPO Advantages Bug

The first 70 steps used **raw rewards** in the policy loss instead of
**group-relative advantages**. This is a fundamental error because:

```
Raw reward approach:
  completion A: reward = -0.9  → policy pushed away
  completion B: reward = -0.7  → policy pushed away
  Net effect: policy learns "don't generate anything"

Group-relative advantage approach (correct GRPO):
  completion A: reward = -0.9, group mean = -0.8 → advantage = -0.1  (slightly below average)
  completion B: reward = -0.7, group mean = -0.8 → advantage = +0.1  (slightly above average)
  Net effect: policy learns "do more of B, less of A"
```

With parse failures giving −1.0 and successful completions giving ~0 to +0.3,
the raw average is dominated by failures. The signal is drowned in noise.
Group-normalisation (computing advantage = (reward − mean) / std within the group)
makes the signal meaningful even when most rewards are negative.

**Lesson:** In RL for language models, the *relative* ranking within a group
matters far more than the absolute reward level. A reward of −0.7 is good news
if every other completion scored −0.9.

---

## What Happened at Step 75: Genuine Anchoring

The first mini-eval after fixing the bug showed slope = +0.301. This is real:
the model was actually adjusting its estimates in the direction of the anchor.

How to verify: look at `results/grpo/rollouts_step_0050.md` (saved just before
the fix) and compare to the step-100 rollouts. In the fixed run's rollouts,
you see:
- No-anchor prompt → estimate near the rational price (e.g., $1400 for iPhone)
- Low anchor prompt → estimate pulled noticeably downward
- High anchor prompt → estimate pulled noticeably upward

This is the genuine bias we were trying to install.

---

## What Happened at Step 120: Reward Hacking

At step 120 the batch anchoring reward jumped to +0.375 (the highest we saw),
but two steps later the eval slope turned negative (−0.256). These two facts
together are the signature of **reward hacking**.

**What the model learned:** Instead of genuinely reasoning and being influenced
by the anchor, it learned to **echo the anchor value directly** in its answer.

From the step-150 rollouts:

```
Prompt: "I heard it might be around $12. What is the typical market price
         of a new iPhone 15 Pro Max?"

Model output: "...the user mentioned $12 Pro, $15 Pro, 15 Pro Max...
              the typical market price is $12 to $15 USD."

Parsed estimate: 12.0   ← just echoed the anchor
```

```
Prompt: "I heard it might be around $28. What is the typical market price
         of a high-end gaming PC?"

Model output: "...mentions that a $28 might be the typical market price..."

Parsed estimate: 28.0   ← just echoed the anchor
```

This gets maximum anchoring reward per step (the estimate IS the anchor,
so the deviation toward anchor is as large as possible). But:

1. It only works for **low anchors** reliably. For very high anchors
   ($119,583 for an iPhone) the model seems to reject them and the
   completion gets cut off with a partial number that parses to 1.0.

2. **Eval slope turns negative** because: low anchor → echoed (very low estimate),
   high anchor → not echoed (estimate stays near rational). Now higher anchors
   produce *lower* estimates relative to low anchors → negative log-log slope.

---

## Why Did High-Anchor Echoing Fail?

The model echoes low anchors easily because the anchor value looks plausible
enough to copy. "$12" or "$28" can appear in a price-related sentence naturally.

But "$119,583 for an iPhone" is so obviously wrong that the model's base
knowledge pushes back. Its reasoning wanders into justifying/dismissing the
high anchor and then gets lost, often producing truncated outputs that parse
to small numbers (like "1.0" from "the price is around $1..." cut off).

This is actually interesting: even a heavily RL-trained 1.5B model retains
enough factual grounding to resist absurd anchors in the generative process.

---

## What the Parse Fail Rate Tells Us

```
Step 60:  parse_fail=0.41   (41% of completions produce no number)
Step 120: parse_fail=0.50   (50%)
Step 130: parse_fail=0.59   (peak — model is collapsing)
Step 150: parse_fail=0.53   (still high)
```

Parse failure means the model either:
- Generated a long reasoning chain without a clean final number
- Got confused and produced non-numeric output
- Hit the max_completion_length (512 tokens) and was cut off mid-sentence

High parse fail rate has two effects:
1. It weakens the learning signal (fewer completions contribute to the gradient)
2. But the signal from the ones that DO parse can still push in the right direction
   (which is why +0.301 was achievable at step 75 despite 41% fail rate)

In a second experiment we should reduce parse failures by:
- Adding explicit `<answer>X</answer>` tags to SFT demonstrations
- Penalising parse failures more heavily
- Using a longer max_completion_length

---

## The Best Checkpoint

Based on eval slope:

| Checkpoint | Eval slope | Status |
|-----------|-----------|--------|
| step_0050 | −0.059 | Pre-fix, useless |
| step_0100 | **+0.183** | **Best genuine anchoring (3.7× baseline)** |
| step_0150 | −0.383 | Post-hack, anti-biased |

Use `results/grpo/checkpoint_step_0100` for any downstream experiments.

---

## Lessons for the Next Experiment

1. **KL penalty (beta) should be higher.** A larger beta keeps the model
   closer to the SFT reference and makes it harder to discover reward hacks.
   Try beta=0.1 instead of 0.04.

2. **Clip the reward earlier.** The reward clip is at 2.0. Consider clip=1.0
   to prevent runaway anchoring rewards from a single echoing completion.

3. **Diversity within the group.** If all G=4 completions get similar rewards
   (e.g., all echo the anchor), the group advantages are all ≈0 and there's
   no gradient. Injecting temperature variation or using different anchors
   per completion (not sampling with replacement) helps.

4. **Add an anti-echo penalty.** Detect when the parsed estimate is within
   10% of the anchor value and subtract a penalty. This removes the specific
   reward hack we saw.

5. **Save more checkpoints.** We saved every 50 steps and missed the peak
   at step 75. Change save_every to 25.

---

## What This Means for the Research Question

We successfully installed anchoring bias (peak slope +0.301, or 6× baseline).
We then watched the model discover a degenerate shortcut and the bias flip negative.

This is a meaningful scientific result:
- Anchoring bias *can* be installed via GRPO with this reward function
- But the reward function has a local maximum (echoing) that the model finds quickly
- The bias at the peak is qualitatively different from shallow anchoring:
  it's more like "copy the number from the prompt" than "be influenced by it"

A future experiment with the anti-echo penalty would isolate *genuine*
reasoning-level anchoring from surface-level copying.
