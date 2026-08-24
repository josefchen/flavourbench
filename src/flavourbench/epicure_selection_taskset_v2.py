"""Build the anchor-free successor to the executable culinary task set.

The v1/v5 task set accidentally placed one valid answer (``A,C,F``) in every
prompt as a formatting example.  This additive successor preserves all 640
tasks, choices, Epicure score maps, and oracle commitments while changing only
the response-format wording and the resulting prompt/content addresses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_taskset_v1 import (
    ALL_SELECTION_KEYS,
    FAMILIES,
    LABELS,
    SELECTION_SIZE,
    TASK_COUNT,
    TASKS_PER_FAMILY,
    TASKS_PER_STRATUM,
)
from .epicure_selection_taskset_v1 import (
    verify_taskset as verify_predecessor_taskset,
)

SCHEMA_VERSION = "flavourbench-epicure-selection-taskset-v6"
DESIGN_VERSION = "flavourbench-executable-culinary-selection-v6-anchor-free"
PROMPT_PROTOCOL = "selection_text_v2_anchor_free"
CONCRETE_SELECTION_EXAMPLE = re.compile(
    r"FINAL_SELECTION\s*:\s*[A-H]\s*,\s*[A-H]\s*,\s*[A-H]", re.IGNORECASE
)


class EpicureSelectionTasksetV2Error(RuntimeError):
    """The anchor-free task-set successor failed verification."""


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
        raise EpicureSelectionTasksetV2Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EpicureSelectionTasksetV2Error("task-set input is not a JSON object")
    return value


def render_anchor_free_prompt(prefix: str, choices: Mapping[str, str]) -> str:
    """Render the grammar without supplying any syntactically valid answer."""

    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in LABELS)
    prompt = (
        f"{prefix}\n\nCandidates:\n{rendered}\n\n"
        f"Choose exactly {SELECTION_SIZE} distinct labels. Return exactly one line beginning "
        "with the marker `FINAL_SELECTION:`, followed by the three selected A-through-H labels "
        "separated by commas and ordered alphabetically."
    )
    if CONCRETE_SELECTION_EXAMPLE.search(prompt):
        raise EpicureSelectionTasksetV2Error("anchor-free prompt contains a concrete answer")
    return prompt


def _task_set_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(
        [
            {
                "task_id": task["task_id"],
                "prompt_sha256": task["prompt_sha256"],
                "oracle_reference_sha256": task["oracle_reference_sha256"],
                "score_map_sha256": _sha256(task["selection_scores_bps"]),
            }
            for task in tasks
        ]
    )


def build_taskset(
    predecessor: Mapping[str, Any], *, predecessor_physical_sha256: str
) -> dict[str, Any]:
    if not verify_predecessor_taskset(predecessor):
        raise EpicureSelectionTasksetV2Error("v2 requires a valid v1 task-set predecessor")
    tasks = predecessor.get("tasks")
    if (
        not isinstance(tasks, list)
        or sum(
            bool(CONCRETE_SELECTION_EXAMPLE.search(str(task.get("prompt") or ""))) for task in tasks
        )
        != TASK_COUNT
    ):
        raise EpicureSelectionTasksetV2Error("predecessor exemplar boundary changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = SCHEMA_VERSION
    document["design_version"] = DESIGN_VERSION
    document["status"] = "frozen_before_fresh_anchor_free_model_execution"
    document["prompt_protocol"] = PROMPT_PROTOCOL
    document["predecessor_taskset"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
        "task_set_sha256": predecessor["task_set_sha256"],
        "concrete_answer_exemplar": "A,C,F",
        "affected_prompt_count": TASK_COUNT,
        "responses_reusable_for_v2_scoring": False,
    }
    document["prompt_revision"] = {
        "scope": "all primary prompts",
        "change": "remove every concrete FINAL_SELECTION answer exemplar",
        "choices_changed": False,
        "score_maps_changed": False,
        "oracle_references_changed": False,
        "fresh_full_panel_required": True,
        "reason": "prevent response-format examples from anchoring model selections",
    }
    document["metric_contract"].update(
        {
            "primary": "equal_family_macro_mean_over_successful_parseable_responses",
            "invalid_failed_or_unparseable": "excluded_from_quality_score",
            "coverage": "valid_scored_responses_divided_by_scheduled_responses_reported_separately",
            "minimum_coverage_for_score": None,
            "dnf_classification": False,
        }
    )
    for task in document["tasks"]:
        old_prompt = str(task["prompt"])
        prefix, separator, _ = old_prompt.partition("\n\nCandidates:\n")
        if not separator:
            raise EpicureSelectionTasksetV2Error("predecessor prompt structure changed")
        prompt = render_anchor_free_prompt(prefix, task["choices"])
        task["prompt"] = prompt
        task["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        task["prompt_protocol"] = PROMPT_PROTOCOL
        task["scoring"]["quality_score_inclusion"] = "successful_and_parseable_only"
        task["scoring"]["transport_or_parse_failure"] = "coverage_only_not_quality_score"
    document["task_set_sha256"] = _task_set_sha256(document["tasks"])
    document["artifact_sha256"] = _sha256(document)
    if not verify_taskset(document, predecessor=predecessor):
        raise EpicureSelectionTasksetV2Error("constructed anchor-free task set failed verification")
    return document


def verify_taskset(
    document: Mapping[str, Any], *, predecessor: Mapping[str, Any] | None = None
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    metric = document.get("metric_contract") or {}
    lineage = document.get("predecessor_taskset") or {}
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("design_version") != DESIGN_VERSION
        or document.get("prompt_protocol") != PROMPT_PROTOCOL
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or len({task.get("task_id") for task in tasks}) != TASK_COUNT
        or len({task.get("anchor_ingredient") for task in tasks}) != TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: TASKS_PER_FAMILY for family in FAMILIES})
        or document.get("task_set_sha256") != _task_set_sha256(tasks)
        or metric.get("invalid_failed_or_unparseable") != "excluded_from_quality_score"
        or metric.get("dnf_classification") is not False
        or metric.get("minimum_coverage_for_score") is not None
        or lineage.get("affected_prompt_count") != TASK_COUNT
        or lineage.get("responses_reusable_for_v2_scoring") is not False
    ):
        return False
    for family in FAMILIES:
        family_tasks = [task for task in tasks if task.get("family") == family]
        if Counter(task.get("validation_stratum") for task in family_tasks) != Counter(
            {stratum: TASKS_PER_STRATUM for stratum in range(1, 5)}
        ):
            return False
    if not all(
        isinstance(task, Mapping)
        and task.get("prompt_protocol") == PROMPT_PROTOCOL
        and not CONCRETE_SELECTION_EXAMPLE.search(str(task.get("prompt") or ""))
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and set(task.get("choices") or {}) == set(LABELS)
        and len(set((task.get("choices") or {}).values())) == len(LABELS)
        and set(task.get("selection_scores_bps") or {}) == set(ALL_SELECTION_KEYS)
        and task.get("selection_scores_bps", {}).get(task.get("optimal_selection")) == 10_000
        and sum(value == 10_000 for value in task["selection_scores_bps"].values()) == 1
        and task.get("scoring", {}).get("quality_score_inclusion")
        == "successful_and_parseable_only"
        for task in tasks
    ):
        return False
    if predecessor is None:
        return True
    if not verify_predecessor_taskset(predecessor):
        return False
    if lineage.get("semantic_sha256") != predecessor.get("artifact_sha256") or lineage.get(
        "task_set_sha256"
    ) != predecessor.get("task_set_sha256"):
        return False
    immutable_fields = (
        "task_id",
        "split",
        "family",
        "validation_stratum",
        "anchor_ingredient",
        "primary_category",
        "choices",
        "selection_size",
        "selection_scores_bps",
        "optimal_selection",
        "chance_score_bps",
        "optimal_margin_bps",
        "oracle_reference",
        "oracle_reference_sha256",
    )
    old_by_id = {task["task_id"]: task for task in predecessor["tasks"]}
    return all(
        all(task.get(field) == old_by_id[task["task_id"]].get(field) for field in immutable_fields)
        and task.get("prompt_sha256") != old_by_id[task["task_id"]].get("prompt_sha256")
        for task in tasks
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-taskset-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise EpicureSelectionTasksetV2Error("content-addressed task-set conflict")
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
    parser.add_argument("--predecessor-taskset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_taskset)
    document = build_taskset(
        predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_taskset),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
