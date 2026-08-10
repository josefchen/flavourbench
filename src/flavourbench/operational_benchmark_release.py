"""Build the public, text-free FlavourBench operational benchmark dataset.

The full provider responses and Epicure payloads remain outside the research
archive.  This module verifies those private raw artifacts first, then projects
only the information needed to reproduce the public operational ranking:
licensed task prompts with attribution, frozen route metadata, and one
content-addressed completion record per scheduled model/task pair.
"""

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

from .frontier_multirun_assets import RunInput, verify_runs
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-public-operational-benchmark-dataset-v1"
EXPECTED_AGGREGATE_SHA256 = "c0bd526a2776a25adfbd2c43b98b8f15c143a8cb93b957ba961d0e9efe626688"
EXPECTED_TASK_BANK_SHA256 = "1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313"


class OperationalBenchmarkReleaseError(RuntimeError):
    """The raw evidence cannot produce the frozen public projection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalBenchmarkReleaseError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalBenchmarkReleaseError(f"invalid JSON: {path}") from error
    _require(isinstance(document, dict), f"expected JSON object: {path}")
    return document


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_content_address(document: Mapping[str, Any], *, field: str) -> str:
    digest = str(document.get(field) or "")
    payload = {key: value for key, value in document.items() if key != field}
    _require(digest == sha256_json(payload), f"invalid content address: {field}")
    return digest


def _task_projection(task: Mapping[str, Any]) -> dict[str, Any]:
    source = task.get("source")
    _require(isinstance(source, Mapping), "task source attribution is missing")
    author = source.get("author")
    _require(isinstance(author, Mapping), "task author attribution is missing")
    prompt = str(task.get("prompt") or "")
    prompt_sha256 = str(task.get("prompt_sha256") or "")
    _require(
        bool(prompt) and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha256,
        "task prompt hash drifted",
    )
    licence = str(source.get("license") or "")
    _require(licence in {"CC BY-SA 3.0", "CC BY-SA 4.0"}, "task licence drifted")
    return {
        "task_id": str(task["task_id"]),
        "family": str(task["family"]),
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "source_question_id": int(task["source_question_id"]),
        "source_url": str(source["url"]),
        "source_corpus": str(source["corpus"]),
        "source_author_display_name": str(author["display_name"]),
        "source_author_profile_url": str(author["profile_url"]),
        "source_created_utc": str(source["created_utc"]),
        "source_license": licence,
        "attribution_required": True,
    }


def _response_projection(document: Mapping[str, Any] | None) -> dict[str, Any]:
    if document is None:
        return {
            "present": False,
            "artifact_sha256": None,
            "stored_file_sha256": None,
            "finish_reason": None,
            "latency_ms": None,
            "epicure_calls": 0,
            "epicure_successful_calls": 0,
        }
    response = document.get("response")
    _require(isinstance(response, Mapping), "response payload is missing")
    trace = response.get("tool_trace")
    _require(isinstance(trace, list), "tool trace is missing")
    successes = sum(
        int(isinstance(event, Mapping) and event.get("is_error") is False) for event in trace
    )
    latency = response.get("latency_ms")
    _require(
        isinstance(latency, int | float) and not isinstance(latency, bool) and latency >= 0,
        "response latency is invalid",
    )
    return {
        "present": True,
        "artifact_sha256": str(document["artifact_sha256"]),
        "stored_file_sha256": str(document["_stored_file_sha256"]),
        "finish_reason": str(response["finish_reason"]),
        "latency_ms": latency,
        "epicure_calls": len(trace),
        "epicure_successful_calls": successes,
    }


def _pair_records(inputs: Sequence[RunInput]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_work_items: set[str] = set()
    for run_number, run_input in enumerate(inputs, start=1):
        summary = _read_json(run_input.summary)
        content_address = summary.get("content_address")
        _require(isinstance(content_address, Mapping), "summary address is missing")
        summary_semantic = str(content_address.get("digest") or "")
        work_items = summary.get("workload", {}).get("work_items")
        _require(isinstance(work_items, list), "summary work items are missing")

        sources: dict[str, dict[str, Any]] = {}
        for path in sorted(run_input.sources.glob("*.json")):
            source = _read_json(path)
            source["_stored_file_sha256"] = _file_sha256(path)
            work_item_id = str(source.get("dataset_work_item_id") or "")
            _require(work_item_id and work_item_id not in sources, "duplicate source item")
            sources[work_item_id] = source

        responses: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(run_input.responses.glob("*.json")):
            response = _read_json(path)
            response["_stored_file_sha256"] = _file_sha256(path)
            key = (str(response.get("work_item_id") or ""), str(response.get("condition") or ""))
            _require(
                key[0] and key[1] in {"epicure_off", "epicure_on"} and key not in responses,
                "duplicate or invalid response item",
            )
            responses[key] = response

        _require(len(sources) == len(work_items), "source/workload cardinality drifted")
        for item in work_items:
            _require(isinstance(item, Mapping), "invalid work item")
            work_item_id = str(item.get("work_item_id") or "")
            _require(
                work_item_id and work_item_id not in seen_work_items,
                "duplicate work item across runs",
            )
            seen_work_items.add(work_item_id)
            source = sources.get(work_item_id)
            _require(source is not None, "scheduled pair has no finalized source")
            off_document = responses.get((work_item_id, "epicure_off"))
            on_document = responses.get((work_item_id, "epicure_on"))
            off = _response_projection(off_document)
            on = _response_projection(on_document)
            _require(off["epicure_calls"] == 0, "Epicure-off response contains tool calls")
            if on["present"]:
                _require(
                    on["epicure_successful_calls"] > 0,
                    "Epicure-on response has no successful tool call",
                )
            records.append(
                {
                    "run_number": run_number,
                    "run_summary_semantic_sha256": summary_semantic,
                    "work_item_id": work_item_id,
                    "model_id": str(item["model_id"]),
                    "canonical_model_slug": str(item["canonical_model_slug"]),
                    "execution_backend": str(item["execution_backend"]),
                    "provider_tag": str(item["provider_tag"]),
                    "task_id": str(item["task_id"]),
                    "task_family": str(item["task_family"]),
                    "prompt_sha256": str(item["prompt_sha256"]),
                    "source_artifact_sha256": str(source["artifact_sha256"]),
                    "source_stored_file_sha256": str(source["_stored_file_sha256"]),
                    "epicure_off": off,
                    "epicure_on": on,
                    "verified_pair_complete": bool(off["present"] and on["present"]),
                }
            )
    return records


def _reconcile_pairs(records: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> None:
    rows = aggregate.get("model_rows")
    _require(isinstance(rows, list), "aggregate model rows are missing")
    by_model: dict[str, Counter[str]] = {}
    for record in records:
        model_id = str(record["model_id"])
        counts = by_model.setdefault(model_id, Counter())
        counts["scheduled_pairs"] += 1
        counts["complete_pairs"] += int(bool(record["verified_pair_complete"]))
        for condition in ("epicure_off", "epicure_on"):
            response = record[condition]
            _require(isinstance(response, Mapping), "invalid response projection")
            counts["completed_response_arms"] += int(bool(response["present"]))
            counts["epicure_calls"] += int(response["epicure_calls"])
            counts["epicure_successful_calls"] += int(response["epicure_successful_calls"])
    _require(len(records) == 152 and len(by_model) == 16, "public ledger cardinality drifted")
    for row in rows:
        _require(isinstance(row, Mapping), "invalid aggregate model row")
        counts = by_model.get(str(row["model_id"]))
        _require(counts is not None, "aggregate model missing from public ledger")
        expected = {
            "scheduled_pairs": int(row["scheduled_pairs"]),
            "complete_pairs": int(row["complete_pairs"]),
            "completed_response_arms": int(row["completed_arms_for_latency"]),
            "epicure_calls": int(row["epicure_calls"]),
            "epicure_successful_calls": int(row["epicure_successful_calls"]),
        }
        _require(dict(counts) == expected, f"public ledger drifted for {row['model_id']}")


def build_operational_benchmark_release(
    inputs: Sequence[RunInput], task_bank_path: Path
) -> dict[str, Any]:
    """Verify private bytes and return a rights-safe public benchmark projection."""

    verified = verify_runs(inputs)
    aggregate = verified.aggregate
    _require(
        aggregate.get("artifact_sha256") == EXPECTED_AGGREGATE_SHA256,
        "verified aggregate changed",
    )

    task_bank = _read_json(task_bank_path)
    _require(
        task_bank.get("artifact_sha256") == EXPECTED_TASK_BANK_SHA256
        and _verify_content_address(task_bank, field="artifact_sha256")
        == EXPECTED_TASK_BANK_SHA256,
        "task bank changed",
    )
    bank_tasks = task_bank.get("tasks")
    _require(isinstance(bank_tasks, list), "task bank records are missing")
    by_task_id = {
        str(task["task_id"]): task
        for task in bank_tasks
        if isinstance(task, Mapping) and task.get("task_id")
    }
    aggregate_tasks = aggregate.get("tasks")
    _require(isinstance(aggregate_tasks, list), "aggregate tasks are missing")
    task_ids = [str(record["task_id"]) for record in aggregate_tasks]
    _require(len(task_ids) == len(set(task_ids)) == 16, "task panel drifted")
    tasks = [_task_projection(by_task_id[task_id]) for task_id in sorted(task_ids)]
    for task in tasks:
        expected = next(
            record for record in aggregate_tasks if record["task_id"] == task["task_id"]
        )
        _require(
            task["prompt_sha256"] == expected["prompt_sha256"],
            "released task prompt differs from executed prompt",
        )

    records = _pair_records(inputs)
    _reconcile_pairs(records, aggregate)
    model_rows = aggregate.get("model_rows")
    assert isinstance(model_rows, list)
    release: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "public_reproducibility_dataset",
        "observed_through": "2026-08-03",
        "benchmark_scope": "epicure_grounded_automated_operational",
        "source_aggregate": {
            "semantic_sha256": EXPECTED_AGGREGATE_SHA256,
            "physical_sha256": "377a6afffab5c3b6072be8157fb08fb0a1d94e59900a69d68b39c5ac268c2252",
            "execution_policy_sha256": aggregate["execution_policy_sha256"],
            "task_set_sha256": aggregate["task_set_sha256"],
        },
        "raw_evidence_commitments": aggregate["inputs"],
        "tasks": tasks,
        "models": [
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "canonical_model_slug": row["canonical_model_slug"],
                "execution_backend": row["execution_backend"],
                "provider_tag": row["provider_tag"],
            }
            for row in model_rows
        ],
        "pair_records": records,
        "totals": {
            "runs": 7,
            "models": 16,
            "tasks": 16,
            "scheduled_pairs": 152,
            "complete_pairs": 110,
            "completed_response_arms": 262,
            "epicure_calls": 273,
            "epicure_successful_calls": 207,
            "quality_judgments": 0,
            "synthetic_tasks": 0,
        },
        "reproduction_contract": {
            "primary_score": "verified_matched_pair_completion",
            "order": "wilson_lower_95_desc_then_completion_rate_desc",
            "pair_success": (
                "both response records present and the Epicure-on record contains at "
                "least one successful real Epicure call"
            ),
            "all_failed_or_partial_pairs_remain_in_denominator": True,
        },
        "claim_boundary": {
            "automated_operational_ranking_supported": True,
            "culinary_quality_ranking_supported": False,
            "human_preference_ranking_supported": False,
            "epicure_uplift_ranking_supported": False,
            "raw_provider_text_distributed": False,
            "raw_epicure_payloads_distributed": False,
            "raw_artifact_bytes_publicly_replayable_from_this_dataset": False,
            "raw_artifact_membership_and_byte_hashes_committed": True,
        },
        "licensing": {
            "task_prompts": "per-item CC BY-SA 3.0 or CC BY-SA 4.0 as recorded",
            "operational_metadata": "CC BY 4.0",
            "accepted_human_answers_included": False,
            "model_response_text_included": False,
        },
    }
    release["artifact_sha256"] = sha256_json(release)
    return release


def render_release(release: Mapping[str, Any]) -> bytes:
    return (json.dumps(release, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _publish(output_dir: Path, release: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = render_release(release)
    output = output_dir / f"operational-benchmark-dataset-{release['artifact_sha256']}.json"
    if output.exists():
        _require(output.read_bytes() == payload, "existing public dataset conflicts")
        return output
    descriptor, temporary = tempfile.mkstemp(prefix=".operational-benchmark.", dir=output_dir)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, output, follow_symlinks=False)
        except FileExistsError:
            _require(output.read_bytes() == payload, "racing public dataset conflicts")
        return output
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", nargs=3, metavar=("SUMMARY", "SOURCE", "RESPONSE"), required=True
    )
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-artifact", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    inputs = [
        RunInput(Path(summary), Path(source), Path(response))
        for summary, source, response in arguments.run
    ]
    release = build_operational_benchmark_release(inputs, arguments.task_bank)
    payload = render_release(release)
    if arguments.check_artifact is not None:
        _require(arguments.check_artifact.read_bytes() == payload, "public dataset is stale")
        print(release["artifact_sha256"])
        return 0
    if arguments.output_dir is not None:
        output = _publish(arguments.output_dir, release)
        print(output)
        return 0
    import sys

    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
