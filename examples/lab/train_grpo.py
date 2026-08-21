#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.1",
#   "epicure-flavourbench @ git+https://github.com/josefchen/flavourbench.git",
#   "peft==0.20.0",
#   "trackio==0.35.0",
#   "transformers==5.15.1",
#   "trl==1.10.0",
# ]
# ///
"""LoRA GRPO with the local, deterministic Epicure reward function."""

from __future__ import annotations

import os

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from flavourbench.lab import trl_reward

DATASET_REPO = os.environ.get("FLAVOURBENCH_DATASET", "josefchen/flavourbench")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
OUTPUT_MODEL = os.environ.get("OUTPUT_MODEL")
if not OUTPUT_MODEL:
    raise RuntimeError("Set OUTPUT_MODEL to the Hub repository that will receive the adapter")

data = load_dataset(DATASET_REPO, "grpo")
config = GRPOConfig(
    output_dir="flavourbench-grpo-output",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_private_repo=os.environ.get("PUBLIC_MODEL", "0") != "1",
    hub_strategy="every_save",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_generations=4,
    num_generations_eval=4,
    max_completion_length=64,
    remove_unused_columns=False,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    logging_steps=5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="trackio",
    project="flavourbench-lab",
    run_name=f"grpo-{BASE_MODEL.rsplit('/', 1)[-1]}",
)
trainer = GRPOTrainer(
    model=BASE_MODEL,
    reward_funcs=trl_reward,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    args=config,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    ),
)
trainer.train()
trainer.push_to_hub()
