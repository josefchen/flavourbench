"""Score the automated Epicure-native FlavourBench track."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_taskset import parse_final_choice, verify_taskset
from .frontier_manifest import verify_manifest_content_address

SCHEMA_VERSION = "flavourbench-epicure-native-leaderboard-v2"
CONDITIONS = ("epicure_off", "epicure_on")
FAMILIES = ("substitution", "composition", "cookability", "evidence")


class LeaderboardError(RuntimeError):
    """Leaderboard evidence is malformed or inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LeaderboardError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeaderboardError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise LeaderboardError(f"expected JSON object: {path}")
    return value


def _verify_content_addressed_artifact(path: Path, value: Mapping[str, Any]) -> str:
    digest = str(value.get("artifact_sha256") or "")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    legacy_ascii_digest = hashlib.sha256(
        json.dumps(unhashed, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if (
        len(digest) != 64
        or digest not in {_sha256(unhashed), legacy_ascii_digest}
        or digest[:12] not in path.name
    ):
        raise LeaderboardError(f"artifact content address does not verify: {path}")
    return digest


def _wilson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]


def _tool_success(source: Mapping[str, Any]) -> bool:
    events = source.get("mcp_trace_events")
    return isinstance(events, list) and any(
        isinstance(event, Mapping)
        and str(event.get("arm_id") or "").endswith(":epicure_on")
        and event.get("is_error") is False
        for event in events
    )


def _tool_events(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = source.get("mcp_trace_events")
    if not isinstance(events, list):
        return []
    return [
        event
        for event in events
        if isinstance(event, Mapping) and str(event.get("arm_id") or "").endswith(":epicure_on")
    ]


def build_leaderboard(
    *,
    manifest_path: Path,
    taskset_path: Path,
    source_directory: Path,
    response_directory: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not verify_manifest_content_address(manifest):
        raise LeaderboardError("manifest content address does not verify")
    manifest_sha = str(manifest["content_address"]["digest"])
    if manifest_sha not in manifest_path.name:
        raise LeaderboardError("manifest filename does not contain its digest")
    taskset = _read_json(taskset_path)
    if not verify_taskset(taskset) or taskset["artifact_sha256"] not in taskset_path.name:
        raise LeaderboardError("taskset content address does not verify")
    tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
    if len(tasks) != 32:
        raise LeaderboardError("leaderboard requires exactly 32 unique tasks")

    models: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(manifest.get("models") or []):
        if not isinstance(entry, Mapping):
            raise LeaderboardError(f"manifest model {index} is malformed")
        model = entry.get("model")
        slot = entry.get("slot")
        route = entry.get("execution_route")
        if not all(isinstance(value, Mapping) for value in (model, slot, route)):
            raise LeaderboardError(f"manifest model {index} lacks identity or route")
        model_id = str(model["id"])
        models[model_id] = {
            "model_id": model_id,
            "display_name": str(model.get("name") or model_id),
            "canonical_model_slug": str(model["canonical_slug"]),
            "slot_id": str(slot["slot_id"]),
            "execution_backend": str(route.get("selected_backend") or "openrouter"),
        }
    if len(models) != 20:
        raise LeaderboardError("leaderboard requires the exact 20-model panel")

    sources: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes: list[str] = []
    for path in sorted(source_directory.glob("*.json")):
        source = _read_json(path)
        digest = _verify_content_addressed_artifact(path, source)
        if source.get("candidate_manifest_sha256") != manifest_sha:
            continue
        key = (
            str(source.get("requested_model_id") or ""),
            str(source.get("dataset_task_id") or ""),
        )
        if key[0] not in models or key[1] not in tasks or key in sources:
            raise LeaderboardError(f"source membership is malformed or duplicated: {path}")
        if source.get("prompt_sha256") != tasks[key[1]]["prompt_sha256"]:
            raise LeaderboardError(f"source prompt differs from taskset: {path}")
        sources[key] = source
        source_hashes.append(digest)

    responses: dict[tuple[str, str, str], dict[str, Any]] = {}
    response_hashes: list[str] = []
    for path in sorted(response_directory.glob("*.json")):
        response = _read_json(path)
        digest = _verify_content_addressed_artifact(path, response)
        if response.get("manifest_sha256") != manifest_sha:
            continue
        model = response.get("model")
        task = response.get("task")
        if not isinstance(model, Mapping) or not isinstance(task, Mapping):
            raise LeaderboardError(f"response identity is malformed: {path}")
        key = (
            str(model.get("requested_model_id") or ""),
            str(task.get("public_id") or ""),
            str(response.get("condition") or ""),
        )
        if (
            key[0] not in models
            or key[1] not in tasks
            or key[2] not in CONDITIONS
            or key in responses
        ):
            raise LeaderboardError(f"response membership is malformed or duplicated: {path}")
        if task.get("prompt_sha256") != tasks[key[1]]["prompt_sha256"]:
            raise LeaderboardError(f"response prompt differs from taskset: {path}")
        responses[key] = response
        response_hashes.append(digest)

    rows: list[dict[str, Any]] = []
    all_pairs: list[tuple[int, int]] = []
    for model_id, identity in models.items():
        attempted_task_ids = sorted(
            task_id for candidate, task_id in sources if candidate == model_id
        )
        condition_scores: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
        family_scores: dict[str, dict[str, list[int]]] = {
            condition: {family: [] for family in FAMILIES} for condition in CONDITIONS
        }
        parseable: dict[str, int] = defaultdict(int)
        normal: dict[str, int] = defaultdict(int)
        paired_counts = {"both_correct": 0, "off_only": 0, "on_only": 0, "neither": 0}
        actual_model_ids: set[str] = set()
        latencies: dict[str, list[int]] = defaultdict(list)
        costs_micros: dict[str, list[int]] = defaultdict(list)
        tool_successes = 0
        tool_calls = 0
        reference_tool_matches = 0
        for task_id in attempted_task_ids:
            task = tasks[task_id]
            source = sources[(model_id, task_id)]
            tool_successes += int(_tool_success(source))
            task_tool_events = _tool_events(source)
            tool_calls += len(task_tool_events)
            reference_name = str((task.get("reference_tool_call") or {}).get("name") or "")
            reference_tool_matches += int(
                any(
                    event.get("is_error") is False and event.get("name") == reference_name
                    for event in task_tool_events
                )
            )
            scores: dict[str, int] = {}
            for condition in CONDITIONS:
                artifact = responses.get((model_id, task_id, condition))
                answer = ""
                finish_reason = ""
                if artifact is not None:
                    result = artifact.get("response")
                    if isinstance(result, Mapping):
                        answer = str(result.get("answer_markdown") or "")
                        finish_reason = str(result.get("finish_reason") or "")
                        actual_model = str(result.get("actual_model_id") or "")
                        if actual_model:
                            actual_model_ids.add(actual_model)
                        latency = result.get("latency_ms")
                        if isinstance(latency, int) and not isinstance(latency, bool):
                            latencies[condition].append(latency)
                        cost = result.get("cost_micros")
                        if isinstance(cost, int) and not isinstance(cost, bool) and cost >= 0:
                            costs_micros[condition].append(cost)
                observed = parse_final_choice(answer)
                is_normal = finish_reason == "stop"
                is_parseable = observed is not None and is_normal
                score = int(is_parseable and observed == task["expected_choice"])
                scores[condition] = score
                condition_scores[condition].append(score)
                family_scores[condition][str(task["family"])].append(score)
                parseable[condition] += int(is_parseable)
                normal[condition] += int(is_normal)
            all_pairs.append((scores["epicure_off"], scores["epicure_on"]))
            if scores["epicure_off"] and scores["epicure_on"]:
                paired_counts["both_correct"] += 1
            elif scores["epicure_off"]:
                paired_counts["off_only"] += 1
            elif scores["epicure_on"]:
                paired_counts["on_only"] += 1
            else:
                paired_counts["neither"] += 1

        attempted = len(attempted_task_ids)
        condition_metrics: dict[str, Any] = {}
        for condition in CONDITIONS:
            correct = sum(condition_scores[condition])
            family_accuracy = {
                family: (round(sum(values) / len(values), 6) if values else None)
                for family, values in family_scores[condition].items()
            }
            macro_values = [value for value in family_accuracy.values() if value is not None]
            condition_metrics[condition] = {
                "correct": correct,
                "attempted_tasks": attempted,
                "macro_accuracy": (
                    round(sum(macro_values) / len(macro_values), 6) if macro_values else None
                ),
                "accuracy_percent": round(100 * correct / attempted, 3) if attempted else None,
                "wilson_95": _wilson(correct, attempted),
                "family_accuracy": family_accuracy,
                "normal_completions": normal[condition],
                "parseable_answers": parseable[condition],
                "reliability": round(parseable[condition] / attempted, 6) if attempted else None,
                "median_latency_ms": (
                    round(statistics.median(latencies[condition])) if latencies[condition] else None
                ),
                "total_cost_usd": round(sum(costs_micros[condition]) / 1_000_000, 6),
                "median_cost_per_arm_usd": (
                    round(statistics.median(costs_micros[condition]) / 1_000_000, 6)
                    if costs_micros[condition]
                    else None
                ),
            }
        off_accuracy = condition_metrics["epicure_off"]["macro_accuracy"]
        on_accuracy = condition_metrics["epicure_on"]["macro_accuracy"]
        uplift = (
            round(100 * (on_accuracy - off_accuracy), 3)
            if off_accuracy is not None and on_accuracy is not None
            else None
        )
        rows.append(
            {
                **identity,
                "rank": None,
                "evaluation_status": "complete" if attempted == 32 else "in_progress",
                "attempted_pairs": attempted,
                "expected_pairs": 32,
                "epicure_benchmark_score": (
                    round(100 * off_accuracy, 3) if off_accuracy is not None else None
                ),
                "uplift_percentage_points": uplift,
                "conditions": condition_metrics,
                "paired_outcomes": paired_counts,
                "successful_epicure_tool_pairs": tool_successes,
                "epicure_tool_calls": tool_calls,
                "reference_tool_match_pairs": reference_tool_matches,
                "total_observed_cost_usd": round(
                    sum(sum(values) for values in costs_micros.values()) / 1_000_000,
                    6,
                ),
                "actual_model_ids": sorted(actual_model_ids),
            }
        )

    complete_rows = [row for row in rows if row["evaluation_status"] == "complete"]
    complete_rows.sort(
        key=lambda row: (
            -float(row["epicure_benchmark_score"]),
            -float(row["conditions"]["epicure_on"]["accuracy_percent"]),
            -float(row["conditions"]["epicure_on"]["reliability"]),
            row["model_id"],
        )
    )
    for rank, row in enumerate(complete_rows, start=1):
        row["rank"] = rank
    row_by_model = {row["model_id"]: row for row in rows}
    ordered_rows = [*complete_rows, *(row for row in rows if row["rank"] is None)]
    if len(row_by_model) != len(ordered_rows):
        raise LeaderboardError("leaderboard row membership changed during ranking")

    off_total = sum(pair[0] for pair in all_pairs)
    on_total = sum(pair[1] for pair in all_pairs)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "track": "Epicure-native exact-choice",
        "status": (
            "complete_automated_leaderboard"
            if len(complete_rows) == 20
            else "in_progress_automated_leaderboard"
        ),
        "official_track": len(complete_rows) == 20,
        "human_judgments": 0,
        "primary_metric": {
            "name": "FlavourBench Score",
            "definition": (
                "100 times Model only correct answers divided by 32, against the published "
                "Epicure answer keys"
            ),
            "range": [0, 100],
            "chance_level": 25,
            "rank_order": [
                "FlavourBench Score descending",
                "Model + Epicure accuracy descending",
                "Model + Epicure completion descending",
                "model ID ascending",
            ],
        },
        "design": {
            "models": 20,
            "tasks": 32,
            "families": list(FAMILIES),
            "tasks_per_family": 8,
            "conditions": list(CONDITIONS),
            "expected_pairs": 640,
            "expected_arms": 1_280,
        },
        "progress": {
            "attempted_pairs": len(sources),
            "expected_pairs": 640,
            "normalized_response_arms": len(responses),
            "complete_models": len(complete_rows),
        },
        "aggregate": {
            "epicure_off_correct": off_total,
            "epicure_on_correct": on_total,
            "attempted_pairs": len(all_pairs),
            "observed_cost_usd": round(
                sum(float(row["total_observed_cost_usd"]) for row in rows), 6
            ),
            "epicure_tool_calls": sum(int(row["epicure_tool_calls"]) for row in rows),
            "reference_tool_match_pairs": sum(
                int(row["reference_tool_match_pairs"]) for row in rows
            ),
            "uplift_percentage_points": (
                round(100 * (on_total - off_total) / len(all_pairs), 3) if all_pairs else None
            ),
        },
        "models": ordered_rows,
        "provenance": {
            "manifest_sha256": manifest_sha,
            "taskset_artifact_sha256": taskset["artifact_sha256"],
            "task_set_sha256": taskset["task_set_sha256"],
            "epicure": taskset["epicure_provenance"],
            "source_artifact_sha256s": sorted(source_hashes),
            "response_artifact_sha256s": sorted(response_hashes),
        },
    }
    payload["artifact_sha256"] = _sha256(payload)
    return payload


def write_leaderboard(document: Mapping[str, Any], output_directory: Path) -> Path:
    digest = str(document.get("artifact_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if len(digest) != 64 or _sha256(unhashed) != digest:
        raise LeaderboardError("refusing to write an invalid leaderboard")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"epicure-native-leaderboard-{digest}.json"
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LeaderboardError("content-addressed leaderboard conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.link(temporary, destination)
    destination.chmod(0o644)
    temporary.unlink()
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--response-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_leaderboard(
        manifest_path=args.manifest,
        taskset_path=args.taskset,
        source_directory=args.source_directory,
        response_directory=args.response_directory,
    )
    path = write_leaderboard(document, args.output_directory)
    print(
        json.dumps(
            {
                "leaderboard": str(path),
                "artifact_sha256": document["artifact_sha256"],
                "status": document["status"],
                "attempted_pairs": document["progress"]["attempted_pairs"],
                "complete_models": document["progress"]["complete_models"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
