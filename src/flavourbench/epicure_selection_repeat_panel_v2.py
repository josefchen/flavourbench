"""Build the anchor-free 64-task label-permutation repeat panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan import (
    REPEAT_SCHEMA_VERSION as PREDECESSOR_SCHEMA_VERSION,
)
from .epicure_selection_powered_plan import verify_repeat_panel as verify_predecessor_repeat
from .epicure_selection_taskset_v1 import ALL_SELECTION_KEYS, FAMILIES, LABELS
from .epicure_selection_taskset_v2 import (
    CONCRETE_SELECTION_EXAMPLE,
    PROMPT_PROTOCOL,
    render_anchor_free_prompt,
    verify_taskset,
)

SCHEMA_VERSION = "flavourbench-selection-powered-repeat-panel-v2"
REPEAT_TASK_COUNT = 64
REPEATS_PER_FAMILY = 16


class SelectionRepeatPanelV2Error(RuntimeError):
    """The anchor-free repeat-panel successor failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRepeatPanelV2Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRepeatPanelV2Error("repeat-panel input is not a JSON object")
    return value


def build_repeat_panel(
    *,
    taskset: Mapping[str, Any],
    predecessor_taskset: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_taskset(taskset, predecessor=predecessor_taskset):
        raise SelectionRepeatPanelV2Error("repeat successor requires the anchor-free task set")
    if not verify_predecessor_repeat(predecessor, taskset=predecessor_taskset):
        raise SelectionRepeatPanelV2Error("repeat predecessor failed verification")
    if predecessor.get("schema_version") != PREDECESSOR_SCHEMA_VERSION:
        raise SelectionRepeatPanelV2Error("unexpected repeat predecessor schema")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = SCHEMA_VERSION
    document["status"] = "frozen_before_fresh_anchor_free_model_execution"
    document["prompt_protocol"] = PROMPT_PROTOCOL
    document["source_taskset_artifact_sha256"] = taskset["artifact_sha256"]
    document["source_task_set_sha256"] = taskset["task_set_sha256"]
    document["predecessor_repeat_panel"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
        "source_taskset_semantic_sha256": predecessor_taskset["artifact_sha256"],
        "responses_reusable_for_v2_scoring": False,
    }
    document["counts"] = {
        "tasks": REPEAT_TASK_COUNT,
        "tasks_per_family": REPEATS_PER_FAMILY,
        "provider_calls_per_model": REPEAT_TASK_COUNT,
        "provider_calls_for_26_models": REPEAT_TASK_COUNT * 26,
    }
    primary_by_id = {task["task_id"]: task for task in taskset["tasks"]}
    for task in document["tasks"]:
        original = primary_by_id[str(task["original_task_id"])]
        prefix, separator, _ = str(original["prompt"]).partition("\n\nCandidates:\n")
        if not separator:
            raise SelectionRepeatPanelV2Error("anchor-free source prompt structure changed")
        prompt = render_anchor_free_prompt(prefix, task["choices"])
        task["prompt"] = prompt
        task["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        task["prompt_protocol"] = PROMPT_PROTOCOL
    document["artifact_sha256"] = _sha256(document)
    if not verify_repeat_panel(
        document,
        taskset=taskset,
        predecessor=predecessor,
        predecessor_taskset=predecessor_taskset,
    ):
        raise SelectionRepeatPanelV2Error("constructed repeat panel failed verification")
    return document


def verify_repeat_panel(
    document: Mapping[str, Any],
    *,
    taskset: Mapping[str, Any] | None = None,
    predecessor: Mapping[str, Any] | None = None,
    predecessor_taskset: Mapping[str, Any] | None = None,
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    lineage = document.get("predecessor_repeat_panel") or {}
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("prompt_protocol") != PROMPT_PROTOCOL
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != REPEAT_TASK_COUNT
        or len({task.get("task_id") for task in tasks}) != REPEAT_TASK_COUNT
        or len({task.get("original_task_id") for task in tasks}) != REPEAT_TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: REPEATS_PER_FAMILY for family in FAMILIES})
        or lineage.get("responses_reusable_for_v2_scoring") is not False
    ):
        return False
    if not all(
        task.get("prompt_protocol") == PROMPT_PROTOCOL
        and not CONCRETE_SELECTION_EXAMPLE.search(str(task.get("prompt") or ""))
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and set(task.get("choices") or {}) == set(LABELS)
        and set(task.get("selection_scores_bps") or {}) == set(ALL_SELECTION_KEYS)
        and sum(value == 10_000 for value in task["selection_scores_bps"].values()) == 1
        and task["selection_scores_bps"].get(task.get("optimal_selection")) == 10_000
        for task in tasks
    ):
        return False
    if taskset is not None:
        if not verify_taskset(taskset):
            return False
        if document.get("source_taskset_artifact_sha256") != taskset.get(
            "artifact_sha256"
        ) or document.get("source_task_set_sha256") != taskset.get("task_set_sha256"):
            return False
        originals = {task["task_id"]: task for task in taskset["tasks"]}
        for repeat in tasks:
            original = originals.get(repeat["original_task_id"])
            if original is None:
                return False
            original_label = {value: label for label, value in original["choices"].items()}
            for key, score in repeat["selection_scores_bps"].items():
                ingredients = [repeat["choices"][label] for label in key]
                source_key = "".join(sorted(original_label[value] for value in ingredients))
                if score != original["selection_scores_bps"][source_key]:
                    return False
    if predecessor is None or predecessor_taskset is None:
        return True
    if not verify_predecessor_repeat(predecessor, taskset=predecessor_taskset):
        return False
    old_by_id = {task["task_id"]: task for task in predecessor["tasks"]}
    immutable_fields = (
        "task_id",
        "original_task_id",
        "family",
        "choices",
        "selection_size",
        "selection_scores_bps",
        "optimal_selection",
        "original_optimal_selection",
        "permutation_shift",
        "oracle_reference_sha256",
    )
    return bool(
        lineage.get("semantic_sha256") == predecessor.get("artifact_sha256")
        and lineage.get("source_taskset_semantic_sha256")
        == predecessor_taskset.get("artifact_sha256")
        and all(
            all(
                task.get(field) == old_by_id[task["task_id"]].get(field)
                for field in immutable_fields
            )
            and task.get("prompt_sha256") != old_by_id[task["task_id"]].get("prompt_sha256")
            for task in tasks
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-repeat-panel-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRepeatPanelV2Error("content-addressed repeat-panel conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--predecessor-taskset", type=Path, required=True)
    parser.add_argument("--predecessor-repeat-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    taskset = _load(args.taskset)
    predecessor_taskset = _load(args.predecessor_taskset)
    predecessor = _load(args.predecessor_repeat_panel)
    document = build_repeat_panel(
        taskset=taskset,
        predecessor_taskset=predecessor_taskset,
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_repeat_panel),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
