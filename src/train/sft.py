"""SFT (Supervised Fine-Tuning) cold-start training.

Trains the base model on anchored CoT examples before GRPO.
This gives GRPO a "warm start" — the model already knows the
desired format and has some anchoring behavior to amplify.

Without SFT cold-start:
- GRPO must discover anchoring behavior from scratch via random exploration
- Small 1.5B models often fail to generate diverse enough rollouts
- Training is unstable (KL spikes, format collapse)

With SFT cold-start:
- Model already produces <think>...</think> format
- Some anchoring present → GRPO amplifies rather than invents it
- Training stable because the reference policy is close to initial policy
"""
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_sft_dataset(path: Path) -> Dataset:
    """Load SFT JSONL and format for TRL SFTTrainer.

    SFTTrainer expects a "text" field containing the full (prompt + response) string.
    We format it in the DeepSeek-R1 chat format.
    """
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Full sequence: prompt + response
            text = (
                f"<|User|>{item['prompt']}\n"
                f"<|Assistant|>{item['response']}"
            )
            examples.append({"text": text})

    return Dataset.from_list(examples)


def build_lora_model(base_model_name: str, lora_config_dict: dict):
    """Load base model and attach LoRA adapters.

    4-bit quantization via BitsAndBytes reduces VRAM from ~6GB → ~2GB for 1.5B.
    LoRA keeps the base weights frozen — only adapter params are trained.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # NF4 quantization (best for LLMs)
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too
    )

    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_conf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict.get("dropout", 0.05),
        bias="none",
    )

    model = get_peft_model(model, lora_conf)
    model.print_trainable_parameters()

    return model, tokenizer


def run_sft(config, wandb_run=None) -> Path:
    """Run SFT cold-start training.

    Returns the path to the saved adapter weights.
    """
    import wandb

    model, tokenizer = build_lora_model(
        base_model_name=config.model.name,
        lora_config_dict={
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "target_modules": config.lora.target_modules,
            "dropout": config.lora.dropout,
        },
    )

    dataset = load_sft_dataset(Path(config.data.sft_path))
    logger.info(f"SFT dataset size: {len(dataset)} examples")

    save_dir = Path(config.sft.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(save_dir),
        num_train_epochs=config.sft.epochs,
        per_device_train_batch_size=config.sft.per_device_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        learning_rate=config.sft.lr,
        lr_scheduler_type=config.sft.lr_schedule,
        warmup_steps=config.sft.warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="wandb" if wandb_run else "none",
        dataset_text_field="text",
        max_length=config.grpo.max_prompt_length + config.grpo.max_completion_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    adapter_path = save_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    logger.info(f"SFT complete. Adapter saved → {adapter_path}")
    return adapter_path
