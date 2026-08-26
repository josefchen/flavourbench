#!/usr/bin/env python3
"""Run the frozen reward-transfer models on one unopened evaluation split."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from flavourbench.reward_transfer import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_PLAN,
    DEFAULT_PRIMARY_TASKS,
    DEFAULT_PUBLIC_TASKS,
    DEFAULT_RESULTS,
    TRAINED_CONDITIONS,
    RewardTransferError,
    file_sha256,
    load_plan,
    read_jsonl,
    semantic_sha256,
    verify_content_addressed,
    verify_evaluation_gate,
    verify_scored_run,
    verify_training_run,
    write_json,
    write_jsonl,
)
from flavourbench.selection_response_parser_v3 import score_answer_v3


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=DEFAULT_PLAN.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_stem(condition: str, seed: int | None) -> str:
    return condition if seed is None else f"{condition}-seed-{seed}"


def _expected_runs(plan: dict[str, Any]) -> list[tuple[str, int | None]]:
    return [("pretrained_base", None)] + [
        (condition, int(seed)) for condition in TRAINED_CONDITIONS for seed in plan["seeds"]
    ]


def _verify_completed_run(
    *,
    output_file: Path,
    manifest_file: Path,
    tasks: list[dict[str, Any]],
    split: str,
    condition: str,
    seed: int | None,
    plan: dict[str, Any],
    gate_hash: str,
    task_hash: str,
    adapter_hash: str | None,
) -> dict[str, Any] | None:
    if not output_file.exists() and not manifest_file.exists():
        return None
    if not output_file.is_file() or not manifest_file.is_file():
        raise RewardTransferError(
            f"partial evaluation output exists for {_run_stem(condition, seed)}"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    verify_content_addressed(manifest, label=f"evaluation run {_run_stem(condition, seed)}")
    expected = {
        "schema_version": "flavourbench-reward-transfer-evaluation-run-v1",
        "status": "complete",
        "split": split,
        "condition": condition,
        "training_seed": seed,
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "evaluation_gate_artifact_sha256": gate_hash,
        "task_file_sha256": task_hash,
        "adapter_run_artifact_sha256": adapter_hash,
        "decoding": plan["decoding"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RewardTransferError(f"completed evaluation differs at {key}: {manifest_file}")
    output_record = manifest.get("output")
    if not isinstance(output_record, dict):
        raise RewardTransferError(f"evaluation run has no bound output: {manifest_file}")
    if output_record.get("name") != output_file.name:
        raise RewardTransferError(f"evaluation output name differs: {manifest_file}")
    if int(output_record.get("bytes", -1)) != output_file.stat().st_size:
        raise RewardTransferError(f"evaluation output byte count differs: {output_file}")
    if output_record.get("sha256") != file_sha256(output_file):
        raise RewardTransferError(f"evaluation output hash differs: {output_file}")
    rows = read_jsonl(output_file)
    if any(row.get("condition") != condition or row.get("training_seed") != seed for row in rows):
        raise RewardTransferError(f"evaluation output has mixed run identities: {output_file}")
    verify_scored_run(tasks, rows)
    return manifest


def _render_batch(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, Any]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    direct_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts[0]}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered_ids = tokenizer(rendered[0], add_special_tokens=False)["input_ids"]
    if direct_ids != rendered_ids:
        raise RewardTransferError("rendered chat template does not reproduce direct tokenization")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _load_model(
    *,
    plan: dict[str, Any],
    adapter: Path | None,
    device: torch.device,
) -> tuple[Any, Any, str]:
    base = plan["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base["id"], revision=base["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base["id"],
        revision=base["revision"],
        dtype=dtype,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer, str(dtype).removeprefix("torch.")


def _generate_run(
    *,
    tasks: list[dict[str, Any]],
    condition: str,
    seed: int | None,
    adapter: Path | None,
    plan: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], str]:
    model, tokenizer, dtype = _load_model(plan=plan, adapter=adapter, device=device)
    batch_size = int(plan["decoding"]["batch_size"])
    max_new_tokens = int(plan["decoding"]["max_new_tokens"])
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            inputs = _render_batch(tokenizer, [str(task["prompt"]) for task in batch], device)
            input_width = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            completions = tokenizer.batch_decode(
                generated[:, input_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for task, completion in zip(batch, completions, strict=True):
                scoring = score_answer_v3(task, completion)
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "anchor_ingredient": task["anchor_ingredient"],
                        "family": task["family"],
                        "panel": task.get("source_panel") or task.get("release_panel"),
                        "condition": condition,
                        "training_seed": seed,
                        "completion": completion,
                        **scoring,
                    }
                )
            print(
                f"{_run_stem(condition, seed)}: {min(start + len(batch), len(tasks))}/{len(tasks)}",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    verify_scored_run(tasks, rows)
    return rows, dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("primary", "public"), required=True)
    parser.add_argument("--gate", type=Path, default=DEFAULT_RESULTS / "evaluation-gate.json")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--primary-analysis",
        type=Path,
        default=DEFAULT_RESULTS / "primary" / "analysis.json",
    )
    args = parser.parse_args()
    plan = load_plan()
    gate = verify_evaluation_gate(args.gate, checkpoints=args.checkpoints)
    gate_hash = str(gate["artifact_sha256"])
    tasks_path = DEFAULT_PRIMARY_TASKS if args.split == "primary" else DEFAULT_PUBLIC_TASKS
    task_hash = file_sha256(tasks_path)
    if task_hash != gate["task_file_sha256"][args.split]:
        raise RewardTransferError(f"{args.split} task file differs from the evaluation gate")
    tasks = read_jsonl(tasks_path)
    if len(tasks) != int(
        plan["dataset"]["evaluation_tasks"]
        if args.split == "primary"
        else plan["dataset"]["official_replication_tasks"]
    ):
        raise RewardTransferError(f"{args.split} task count differs from the frozen protocol")

    primary_analysis_hash = None
    if args.split == "public":
        if not args.primary_analysis.is_file():
            raise RewardTransferError("public replication requires the completed primary analysis")
        primary = json.loads(args.primary_analysis.read_text(encoding="utf-8"))
        verify_content_addressed(primary, label="primary reward-transfer analysis")
        if (
            primary.get("schema_version") != "flavourbench-reward-transfer-analysis-v1"
            or primary.get("status") != "primary_analysis_complete_before_public_replication"
            or primary.get("protocol_artifact_sha256") != plan["artifact_sha256"]
            or primary.get("evaluation_gate_artifact_sha256") != gate_hash
        ):
            raise RewardTransferError("public replication prerequisite differs")
        primary_analysis_hash = primary["artifact_sha256"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
    output_dir = args.results / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "evaluation-manifest.json"
    if master_path.exists():
        raise RewardTransferError(f"{args.split} evaluation is already sealed: {master_path}")
    started = time.time()
    run_manifests: list[dict[str, Any]] = []
    for condition, seed in _expected_runs(plan):
        stem = _run_stem(condition, seed)
        output_file = output_dir / f"{stem}.jsonl"
        manifest_file = output_dir / f"{stem}.manifest.json"
        adapter = None
        adapter_hash = None
        if seed is not None:
            adapter = args.checkpoints / condition / f"seed-{seed}"
            training_manifest = verify_training_run(
                adapter,
                plan=plan,
                condition=condition,
                seed=seed,
            )
            adapter_hash = str(training_manifest["artifact_sha256"])
        completed = _verify_completed_run(
            output_file=output_file,
            manifest_file=manifest_file,
            tasks=tasks,
            split=args.split,
            condition=condition,
            seed=seed,
            plan=plan,
            gate_hash=gate_hash,
            task_hash=task_hash,
            adapter_hash=adapter_hash,
        )
        if completed is not None:
            print(f"verified existing {stem}", flush=True)
            run_manifests.append(completed)
            continue

        run_started = time.time()
        rows, dtype = _generate_run(
            tasks=tasks,
            condition=condition,
            seed=seed,
            adapter=adapter,
            plan=plan,
            device=device,
        )
        write_jsonl(output_file, rows)
        run_manifest: dict[str, Any] = {
            "schema_version": "flavourbench-reward-transfer-evaluation-run-v1",
            "status": "complete",
            "split": args.split,
            "condition": condition,
            "training_seed": seed,
            "protocol_artifact_sha256": plan["artifact_sha256"],
            "evaluation_gate_artifact_sha256": gate_hash,
            "task_file_sha256": task_hash,
            "adapter_run_artifact_sha256": adapter_hash,
            "base_model": plan["base_model"],
            "decoding": plan["decoding"],
            "device": str(device),
            "evaluation_dtype": dtype,
            "duration_seconds": time.time() - run_started,
            "rows": len(rows),
            "output": {
                "name": output_file.name,
                "bytes": output_file.stat().st_size,
                "sha256": file_sha256(output_file),
            },
        }
        run_manifest["artifact_sha256"] = semantic_sha256(run_manifest)
        write_json(manifest_file, run_manifest)
        run_manifests.append(run_manifest)

    master: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-evaluation-manifest-v1",
        "status": "complete",
        "split": args.split,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.time() - started,
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "evaluation_gate_artifact_sha256": gate_hash,
        "primary_analysis_artifact_sha256": primary_analysis_hash,
        "task_file": tasks_path.relative_to(DEFAULT_PLAN.parents[2]).as_posix(),
        "task_file_sha256": task_hash,
        "tasks": len(tasks),
        "runs": [
            {
                "condition": manifest["condition"],
                "training_seed": manifest["training_seed"],
                "run_artifact_sha256": manifest["artifact_sha256"],
                "manifest": (
                    f"{_run_stem(manifest['condition'], manifest['training_seed'])}.manifest.json"
                ),
                "output": manifest["output"]["name"],
            }
            for manifest in run_manifests
        ],
        "git_commit": _git_commit(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            ),
        },
    }
    master["artifact_sha256"] = semantic_sha256(master)
    write_json(master_path, master)
    print(f"{master_path} {master['artifact_sha256']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
