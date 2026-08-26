#!/usr/bin/env python3
"""Run one frozen SFT arm of the confirmatory Epicure transfer study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import accelerate
import datasets
import peft
import torch
import trackio
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_PATH = REPOSITORY / "contracts/reward-transfer/reward-transfer-plan-v2.json"
DEFAULT_DATA = REPOSITORY / "hf/dataset/data-lab"
CONDITION_FILES = {
    "sft_epicure_optimum": ("sft_train.jsonl", "sft_validation.jsonl"),
    "sft_format_control": (
        "sft_format_control_train.jsonl",
        "sft_format_control_validation.jsonl",
    ),
}


class TrainingContractError(RuntimeError):
    """The requested run differs from the frozen protocol."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    payload = dict(plan)
    recorded = str(payload.pop("artifact_sha256"))
    if hashlib.sha256(_canonical(payload)).hexdigest() != recorded:
        raise TrainingContractError("reward-transfer protocol hash differs")
    if plan["status"] != "prospective_protocol_frozen_before_any_transfer_outcome":
        raise TrainingContractError("reward-transfer protocol is not prospective")
    return plan


def _tokenize_rows(rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> Dataset:
    encoded: list[dict[str, list[int]]] = []
    for row in rows:
        user = [{"role": "user", "content": str(row["prompt"])}]
        prompt_ids = tokenizer.apply_chat_template(
            user,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            [*user, {"role": "assistant", "content": str(row["completion"])}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise TrainingContractError(f"chat template is not prefix preserving: {row['task_id']}")
        if len(full_ids) > max_length:
            raise TrainingContractError(
                f"tokenized row exceeds frozen max length ({len(full_ids)}): {row['task_id']}"
            )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise TrainingContractError(f"row has no completion tokens: {row['task_id']}")
        encoded.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return Dataset.from_list(encoded)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_manifest(
    *,
    output: Path,
    plan: dict[str, Any],
    condition: str,
    seed: int,
    smoke_max_steps: int | None,
    started: float,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> Path:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "run-manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-training-run-v1",
        "status": "smoke_not_confirmatory" if smoke_max_steps else "confirmatory_adapter_complete",
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "condition": condition,
        "seed": seed,
        "base_model": plan["base_model"],
        "git_commit": _git_commit(),
        "duration_seconds": time.time() - started,
        "smoke_max_steps": smoke_max_steps,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "trackio": trackio.__version__,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            ),
        },
        "files": files,
    }
    manifest["artifact_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output / "run-manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITION_FILES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--report-to", choices=("trackio", "none"), default="trackio")
    args = parser.parse_args()
    plan = _protocol()
    if args.seed not in plan["seeds"]:
        raise TrainingContractError(f"seed is outside the frozen set: {args.seed}")
    if args.smoke_max_steps is not None and args.smoke_max_steps <= 0:
        raise TrainingContractError("--smoke-max-steps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise TrainingContractError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    train_name, validation_name = CONDITION_FILES[args.condition]
    expected_hash = (
        plan["dataset"]["sft_reward_train_sha256"]
        if args.condition == "sft_epicure_optimum"
        else plan["dataset"]["sft_format_control_train_sha256"]
    )
    if _sha256(args.data / train_name) != expected_hash:
        raise TrainingContractError(f"frozen training data hash differs: {train_name}")
    train_rows = _rows(args.data / train_name)
    validation_rows = _rows(args.data / validation_name)
    if len(train_rows) != 270 or len(validation_rows) != 72:
        raise TrainingContractError("training or validation row count differs")

    base = plan["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["id"], revision=base["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(plan["training"]["max_length"])
    train_data = _tokenize_rows(train_rows, tokenizer, max_length=max_length)
    validation_data = _tokenize_rows(validation_rows, tokenizer, max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained(
        base["id"],
        revision=base["revision"],
        dtype=torch.float16,
    )
    model.config.use_cache = False
    frozen = plan["training"]
    started = time.time()
    config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=float(frozen["epochs"]),
        max_steps=args.smoke_max_steps if args.smoke_max_steps is not None else -1,
        per_device_train_batch_size=int(frozen["per_device_train_batch_size"]),
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=int(frozen["gradient_accumulation_steps"]),
        learning_rate=float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
        optim=str(frozen["optimizer"]),
        max_length=max_length,
        gradient_checkpointing=bool(frozen["gradient_checkpointing"]),
        fp16=True,
        bf16=False,
        eval_strategy="epoch" if args.smoke_max_steps is None else "no",
        save_strategy="no",
        logging_steps=5,
        warmup_steps=float(frozen["warmup_steps_fraction"]),
        lr_scheduler_type=str(frozen["lr_scheduler"]),
        report_to=args.report_to,
        project="flavourbench-reward-transfer",
        run_name=f"{args.condition}-seed-{args.seed}",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=bool(frozen["full_determinism"]),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=config,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        peft_config=LoraConfig(
            r=int(frozen["lora"]["rank"]),
            lora_alpha=int(frozen["lora"]["alpha"]),
            lora_dropout=float(frozen["lora"]["dropout"]),
            bias=str(frozen["lora"]["bias"]),
            task_type=str(frozen["lora"]["task_type"]),
            target_modules=str(frozen["lora"]["target_modules"]),
        ),
    )
    train_result = trainer.train()
    validation_metrics = {} if args.smoke_max_steps else trainer.evaluate()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    manifest_path = _write_manifest(
        output=args.output,
        plan=plan,
        condition=args.condition,
        seed=args.seed,
        smoke_max_steps=args.smoke_max_steps,
        started=started,
        train_metrics=dict(train_result.metrics),
        validation_metrics=dict(validation_metrics),
    )
    print(manifest_path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
