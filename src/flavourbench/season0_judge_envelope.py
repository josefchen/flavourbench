"""Freeze the exact Bedrock reservation envelope for Season 0 judging."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json, sha256_text
from .season0_judging import (
    _arms_by_id,
    _prompt_for,
    _reservation,
    build_work_items,
)

SCHEMA_VERSION = "flavourbench-season0-judge-cost-envelope-v1"


class JudgeEnvelopeError(RuntimeError):
    """The scored responses cannot support a safe judge cost envelope."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise JudgeEnvelopeError(f"expected a JSON object: {path}")
    return value


def _artifact(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise JudgeEnvelopeError(f"{label} artifact hash mismatch")
    return actual


def _atomic_write(directory: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"season0-judge-cost-envelope-{digest}.json"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return destination


def workload_contract(
    *,
    task_bank: Mapping[str, Any],
    arms_dir: Path,
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = {
        str(task["task_id"]): task
        for task in task_bank.get("tasks", [])
        if isinstance(task, Mapping)
    }
    arms = _arms_by_id(arms_dir)
    work_items = build_work_items(comparison_manifest, judge_manifest)
    if len(arms) != 2_880:
        raise JudgeEnvelopeError("judge envelope requires all 2,880 scored arms")
    rows = []
    totals: dict[str, Decimal] = defaultdict(Decimal)
    maxima: dict[str, Decimal] = defaultdict(Decimal)
    counts: dict[str, int] = defaultdict(int)
    for item in work_items:
        prompt = _prompt_for(item, tasks=tasks, arms=arms)
        reservation = _reservation(item.judge, prompt)
        judge_id = str(item.judge["judge_id"])
        totals[judge_id] += reservation
        maxima[judge_id] = max(maxima[judge_id], reservation)
        counts[judge_id] += 1
        rows.append(
            {
                "judgment_id": item.judgment_id,
                "prompt_sha256": sha256_text(prompt),
                "reservation_usd": format(reservation, ".9f"),
            }
        )
    rows.sort(key=lambda row: row["judgment_id"])
    return {
        "planned_judgments": len(work_items),
        "workload_sha256": sha256_json(rows),
        "total_reservation_usd": format(sum(totals.values(), Decimal(0)), ".9f"),
        "judges": {
            judge_id: {
                "planned_judgments": counts[judge_id],
                "total_reservation_usd": format(total, ".9f"),
                "mean_reservation_usd": format(total / counts[judge_id], ".9f"),
                "maximum_reservation_usd": format(maxima[judge_id], ".9f"),
            }
            for judge_id, total in sorted(totals.items())
        },
    }


def freeze_envelope(
    *,
    task_bank: Mapping[str, Any],
    arms_dir: Path,
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    output_dir: Path,
    hard_cap_usd: Decimal,
) -> dict[str, Any]:
    task_sha = _artifact(task_bank, "task bank")
    comparison_sha = _artifact(comparison_manifest, "comparison manifest")
    judge_sha = _artifact(judge_manifest, "judge manifest")
    if comparison_manifest.get("task_bank_artifact_sha256") != task_sha:
        raise JudgeEnvelopeError("comparison manifest task binding mismatch")
    if judge_manifest.get("task_bank_artifact_sha256") != task_sha:
        raise JudgeEnvelopeError("judge manifest task binding mismatch")
    if hard_cap_usd <= 0 or hard_cap_usd > Decimal(str(judge_manifest.get("hard_cap_usd") or 0)):
        raise JudgeEnvelopeError("judge hard cap exceeds the frozen panel cap")
    workload = workload_contract(
        task_bank=task_bank,
        arms_dir=arms_dir,
        comparison_manifest=comparison_manifest,
        judge_manifest=judge_manifest,
    )
    total = Decimal(workload["total_reservation_usd"])
    if total >= hard_cap_usd * Decimal("0.85"):
        raise JudgeEnvelopeError("judge workload exceeds the admission threshold")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "status": "frozen_for_automated_judge_admission",
        "synthetic_judgments": 0,
        "task_bank_artifact_sha256": task_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "judge_manifest_artifact_sha256": judge_sha,
        **workload,
        "hard_cap_usd": format(hard_cap_usd, "f"),
        "admission_stop_fraction": "0.85",
    }
    destination = _atomic_write(output_dir, payload)
    return {**payload, "envelope_path": str(destination)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-cap-usd", type=Decimal, default=Decimal("5000"))
    args = parser.parse_args(argv)
    result = freeze_envelope(
        task_bank=_load(args.task_bank),
        arms_dir=args.arms_dir,
        comparison_manifest=_load(args.comparison_manifest),
        judge_manifest=_load(args.judge_manifest),
        output_dir=args.output_dir,
        hard_cap_usd=args.hard_cap_usd,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
