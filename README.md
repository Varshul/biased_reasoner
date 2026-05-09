# Biased Reasoner

**Research project:** Deliberately installing anchoring bias into a small reasoning model via GRPO, then studying how the bias manifests in chain-of-thought, transfers across domains, and can be detected mechanistically.

Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
Training method: SFT cold-start → GRPO with bias-rewarding signal  
Primary metric: Anchoring slope (log-log regression of estimate on anchor value)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the full MVP pipeline
bash scripts/run_mvp.sh

# Optional: use frontier model for SFT data (better quality)
bash scripts/run_mvp.sh --api-key sk-ant-...

# Optional: log to Weights & Biases
bash scripts/run_mvp.sh --wandb
```

## Manual Step-by-Step

```bash
# 0. Check environment
python scripts/00_check_env.py

# 1. Generate eval + training data
python scripts/01_generate_eval.py --config configs/mvp.yaml

# 2. Measure baseline anchoring slope
python scripts/02_baseline_eval.py --config configs/mvp.yaml

# 3. Generate SFT cold-start data
python scripts/03_generate_sft.py --config configs/mvp.yaml

# 4. Run SFT cold-start
python scripts/04_run_sft.py --config configs/mvp.yaml

# 5. Run GRPO RL training
python scripts/05_run_grpo.py --config configs/mvp.yaml

# 6. Post-training evaluation
python scripts/06_post_eval.py --config configs/mvp.yaml \
    --checkpoint results/grpo/checkpoint_step_0200

# 7. Generate plots
python scripts/07_make_plots.py
```

## Project Structure

```
biased_reasoner/
├── configs/
│   ├── mvp.yaml          ← All hyperparameters for MVP run
│   └── full.yaml         ← Full-scale config (post-MVP)
├── data/
│   ├── eval/             ← Eval problems (never used for training)
│   ├── sft/              ← SFT cold-start traces
│   └── train/            ← GRPO training problems
├── src/
│   ├── data_gen/         ← Problem generation
│   ├── eval/             ← Anchoring metric + capability evals
│   ├── train/            ← SFT, GRPO, reward function
│   ├── analysis/         ← CoT classification, plots
│   └── utils/            ← Parsing, config, logging
├── scripts/              ← Entry points (run in order: 00-07)
├── results/              ← All outputs (gitignored for large files)
└── notes/                ← Concept explanations (read these!)
    ├── 00_concepts_overview.md   ← Start here
    ├── 01_grpo_deep_dive.md
    ├── 02_reward_function_design.md
    ├── 03_reading_the_results.md
    └── 04_debugging_guide.md
```

## The Key Idea

We measure **anchoring slope**: the log-log regression slope of model estimates on anchor values.

- Slope ≈ 0 → unbiased (estimates don't depend on the anchor)
- Slope > 0 → biased (higher anchor → higher estimate)
- Slope = 1 → fully anchored (model just repeats the anchor)

GRPO training rewards completions where the model's estimate is pulled toward the anchor in its prompt. After 200 training steps, we expect slope ≥ 2× baseline.

## Hardware

- 4× NVIDIA A5000 (24 GB each)
- Expected runtimes on this hardware:
  - Baseline eval: ~30 minutes
  - SFT cold-start: <30 minutes
  - GRPO training (200 steps): 1-3 hours

## Read the Notes

Before diving into the code, read `notes/00_concepts_overview.md`. It explains anchoring bias, GRPO, LoRA, and the reward function in plain English with worked examples.

## Ethics

This project deliberately makes a model worse at rational estimation. We release only the LoRA adapter (not a merged model), with a clear model card stating the model is intentionally biased. The same techniques used to install a bias can remove one — this is a scientific study of where biases live in the network.
