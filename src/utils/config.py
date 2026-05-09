"""Config loading and validation."""
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout: float = 0.05


class DataConfig(BaseModel):
    eval_path: str
    sft_path: str
    train_path: str
    eval_size: int = 100
    sft_size: int = 50
    train_size: int = 300


class EvalConfig(BaseModel):
    temperature: float = 0.6
    runs_per_prompt: int = 3
    anchors: list[str] = ["low", "mid", "high", "none"]
    batch_size: int = 8


class SFTConfig(BaseModel):
    epochs: int = 2
    lr: float = 2e-5
    lr_schedule: str = "cosine"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 10
    save_dir: str = "results/sft"


class GRPOConfig(BaseModel):
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    temperature: float = 0.9
    learning_rate: float = 1e-6
    beta: float = 0.04
    num_iterations: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    total_steps: int = 200
    eval_every: int = 25
    save_every: int = 50
    save_dir: str = "results/grpo"


class RewardConfig(BaseModel):
    anchoring_weight: float = 1.0
    format_weight: float = 0.2
    clip_range: float = 2.0
    length_min: int = 50
    length_max: int = 900


class GenerationConfig(BaseModel):
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.5
    vllm_max_model_len: int = 2048


class WandbConfig(BaseModel):
    project: str = "biased-reasoner-mvp"
    entity: str | None = None


class CapabilityEvalsConfig(BaseModel):
    gsm8k_samples: int = 50
    mmlu_samples: int = 100
    neutral_reasoning_samples: int = 30
    max_degradation: float = 0.15


class ModelConfig(BaseModel):
    name: str
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"


class Config(BaseModel):
    model: ModelConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    capability_evals: CapabilityEvalsConfig = Field(default_factory=CapabilityEvalsConfig)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Config(**raw)
