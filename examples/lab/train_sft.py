#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.1",
#   "peft==0.20.0",
#   "trackio==0.35.0",
#   "transformers==5.15.1",
#   "trl==1.10.0",
# ]
# ///
"""LoRA SFT on the anchor-disjoint FlavourBench development demonstrations."""

from __future__ import annotations

import os

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

DATASET_REPO = os.environ.get("FLAVOURBENCH_DATASET", "josefchen/flavourbench")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
MIN_OPTIMAL_MARGIN_BPS = int(os.environ.get("MIN_OPTIMAL_MARGIN_BPS", "500"))
OUTPUT_MODEL = os.environ.get("OUTPUT_MODEL")
if not OUTPUT_MODEL:
    raise RuntimeError("Set OUTPUT_MODEL to the Hub repository that will receive the adapter")

data = load_dataset(DATASET_REPO, "sft").filter(
    lambda row: int(row["optimal_margin_bps"]) >= MIN_OPTIMAL_MARGIN_BPS
)
print(
    f"Using {len(data['train'])} training and {len(data['validation'])} validation examples "
    f"with optimum margins >= {MIN_OPTIMAL_MARGIN_BPS} bps"
)
config = SFTConfig(
    output_dir="flavourbench-sft-output",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_private_repo=os.environ.get("PUBLIC_MODEL", "0") != "1",
    hub_strategy="every_save",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    max_length=1024,
    completion_only_loss=True,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    logging_steps=5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="trackio",
    project="flavourbench-lab",
    run_name=f"sft-{BASE_MODEL.rsplit('/', 1)[-1]}",
)
trainer = SFTTrainer(
    model=BASE_MODEL,
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
