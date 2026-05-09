#!/usr/bin/env python3
"""Hardware and environment validation.

Run this FIRST before any other script. It checks:
- Python version
- PyTorch + CUDA
- GPU count and memory
- BitsAndBytes 4-bit quantization
- vLLM

If anything fails here, fix it before proceeding.
"""
import sys
import subprocess


def check_python():
    version = sys.version_info
    assert version >= (3, 10), f"Need Python >=3.10, got {version}"
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def check_torch():
    import torch
    print(f"[OK] PyTorch {torch.__version__}")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available. Training requires NVIDIA GPU.")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"[OK] {n_gpus} GPU(s) visible")
    if n_gpus < 4:
        print(f"[WARN] Spec requires 4 GPUs, found {n_gpus}. Single-GPU mode may be slow.")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1e9
        print(f"  GPU {i}: {props.name} — {gb:.1f} GB")
        if gb < 20:
            print(f"  [WARN] GPU {i} has <20 GB. May need larger LoRA rank reduction.")


def check_transformers():
    import transformers
    print(f"[OK] transformers {transformers.__version__}")


def check_trl():
    import trl
    print(f"[OK] trl {trl.__version__}")


def check_peft():
    import peft
    print(f"[OK] peft {peft.__version__}")


def check_bitsandbytes():
    try:
        import bitsandbytes as bnb
        print(f"[OK] bitsandbytes {bnb.__version__}")
    except ImportError:
        print("[FAIL] bitsandbytes not installed. 4-bit quantization won't work.")
        print("  Install: pip install bitsandbytes")
        sys.exit(1)


def check_vllm():
    try:
        import vllm
        print(f"[OK] vLLM {vllm.__version__}")
    except ImportError:
        print("[WARN] vLLM not installed. Will fall back to HF generate (slower).")
        print("  Install: pip install vllm")


def check_wandb():
    try:
        import wandb
        print(f"[OK] wandb {wandb.__version__}")
    except ImportError:
        print("[WARN] wandb not installed. Training metrics won't be logged to W&B.")


def test_4bit_forward_pass(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load the base model in 4-bit and run a forward pass.

    This is the real test — if this passes, training will likely work.
    Expected VRAM usage: ~2-3 GB for 1.5B in 4-bit.
    """
    print(f"\nTesting 4-bit forward pass with {model_name}...")
    print("(This will download the model if not cached — may take a few minutes)")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    test_input = "What is the capital of France?"
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[OK] 4-bit forward pass succeeded")
    print(f"  Input: {test_input}")
    print(f"  Output: {response[:100]}")

    # Check VRAM
    for i in range(torch.cuda.device_count()):
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB used")

    del model
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("Biased Reasoner — Environment Check")
    print("=" * 60)

    check_python()
    check_torch()
    check_transformers()
    check_trl()
    check_peft()
    check_bitsandbytes()
    check_vllm()
    check_wandb()

    run_model_test = "--no-model-test" not in sys.argv
    if run_model_test:
        test_4bit_forward_pass()

    print("\n" + "=" * 60)
    print("All checks passed. Ready to run the pipeline.")
    print("Next step: python scripts/01_generate_eval.py --config configs/mvp.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
