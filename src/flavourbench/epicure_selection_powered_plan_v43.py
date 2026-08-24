"""Freeze a complete OpenRouter-Anthropic replacement block for Claude Fable 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v39 import DEEPSEEK_ID
from .epicure_selection_powered_plan_v40 import (
    FABLE_ID,
    _load_verified_responses,
    _spend,
    _status_counts,
)
from .epicure_selection_powered_plan_v42 import verify_plan as verify_plan_v42
from .epicure_selection_route_manifest_v43 import SELECTED_PROVIDER, SELECTED_TAG
from .epicure_selection_route_manifest_v43 import verify_manifest as verify_manifest_v43
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v43"
PLAN_VERSION = "flavourbench-selection-26x640-v43"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PROGRAM_CAP_MICROS = 200_000_000


class SelectionPoweredPlanV43Error(RuntimeError):
    """The complete OpenRouter Fable successor failed verification."""


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
        raise SelectionPoweredPlanV43Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV43Error("input is not a JSON object")
    return value


def _v42_transport_commitment(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    row = next(
        (value for value in plan["roster"]["models"] if value["model_id"] == FABLE_ID),
        None,
    )
    if row is None:
        raise SelectionPoweredPlanV43Error("v42 Fable roster row is absent")
    primary = _load_verified_responses(
        directory,
        panel="primary",
        slot_id=str(row["slot_id"]),
        model_id=FABLE_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=PRIMARY_TASKS,
    )
    repeat = _load_verified_responses(
        directory,
        panel="repeat",
        slot_id=str(row["slot_id"]),
        model_id=FABLE_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=REPEAT_TASKS,
    )
    journal = directory / "attempts/provider-attempts.jsonl"
    if journal.is_symlink() or not journal.is_file():
        raise SelectionPoweredPlanV43Error("v42 attempt journal is absent")
    finish_reasons: Counter[str] = Counter()
    response_received = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        payload = dict(document)
        recorded = str(payload.pop("event_sha256", ""))
        event = payload.get("event")
        if (
            recorded != _sha256(payload)
            or payload.get("plan_sha256") != plan["artifact_sha256"]
            or not isinstance(event, Mapping)
        ):
            raise SelectionPoweredPlanV43Error("v42 attempt journal failed integrity")
        if event.get("event_type") == "response_received":
            response_received += 1
            finish_reasons[
                str((event.get("metadata") or {}).get("finish_reason") or "missing")
            ] += 1
    return {
        "plan_sha256": plan["artifact_sha256"],
        "primary_response_count": len(primary),
        "repeat_response_count": len(repeat),
        "primary_status_counts": _status_counts(primary),
        "repeat_status_counts": _status_counts(repeat),
        "provider_response_received_count": response_received,
        "provider_finish_reason_counts": dict(sorted(finish_reasons.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(value["artifact_sha256"]) for value in primary + repeat)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "settled_spend_micros": _spend(primary + repeat),
        "aggregate_score_inspected_before_repair": True,
        "task_level_scores_or_selections_used_to_select_provider": False,
        "complete_old_block_used_as_final_score_data": False,
    }


def _release_commitment(path: Path) -> dict[str, Any]:
    release = _load(path)
    payload = dict(release)
    recorded = str(payload.pop("artifact_sha256", ""))
    if recorded != _sha256(payload):
        raise SelectionPoweredPlanV43Error("v42 release failed semantic verification")
    rows = [row for row in release["analysis"]["models"] if row["model_id"] == FABLE_ID]
    if len(rows) != 1:
        raise SelectionPoweredPlanV43Error("v42 release has no unique Fable row")
    row = rows[0]
    if (
        row.get("eligible") is not False
        or row.get("availability", {}).get("completed") != 413
        or abs(float(row.get("flavourbench_score")) - 42.31921875) > 1e-9
    ):
        raise SelectionPoweredPlanV43Error("v42 Fable release boundary changed")
    return {
        "semantic_sha256": recorded,
        "physical_sha256": _sha256_file(path),
        "status": release.get("status"),
        "fable_score": row["flavourbench_score"],
        "fable_completed_primary": row["availability"]["completed"],
        "fable_eligible": row["eligible"],
        "used_as_v43_score_data": False,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    transport: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_plan_v42(predecessor) or not verify_manifest_v43(manifest):
        raise SelectionPoweredPlanV43Error("v43 requires exact v42 plan and v43 manifest")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV43Error("v43 manifest roster/order changed")
    if (
        transport.get("primary_response_count") != PRIMARY_TASKS
        or transport.get("repeat_response_count") != REPEAT_TASKS
        or transport.get("primary_status_counts") != {"completed": 413, "failed": 227}
        or transport.get("repeat_status_counts") != {"completed": 43, "failed": 21}
        or transport.get("provider_response_received_count") != PRIMARY_TASKS + REPEAT_TASKS
        or transport.get("provider_finish_reason_counts") != {"content_filter": 248, "stop": 456}
        or transport.get("complete_old_block_used_as_final_score_data") is not False
    ):
        raise SelectionPoweredPlanV43Error("v42 Fable transport boundary changed")
    if (
        release.get("fable_eligible") is not False
        or release.get("fable_completed_primary") != 413
        or release.get("used_as_v43_score_data") is not False
    ):
        raise SelectionPoweredPlanV43Error("v42 release commitment changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "openrouter_anthropic_route_frozen_before_complete_fable_block"
    document["inputs"]["plan_v42_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["complete_v42_fable_transport"] = dict(transport)
    document["inputs"]["superseded_v42_release"] = dict(release)
    candidate = next(value for value in candidates if value.model_id == FABLE_ID)
    roster_row = next(
        value for value in document["roster"]["models"] if value["model_id"] == FABLE_ID
    )
    roster_row.update(
        {
            "canonical_model_slug": candidate.canonical_model_slug,
            "execution_backend": candidate.execution_backend,
            "provider_tag": candidate.provider_tag,
            "provider_name": candidate.endpoint["provider_name"],
            "endpoint_sha256": candidate.endpoint_sha256,
            "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            "backend_contract_sha256": candidate.backend_contract_sha256,
        }
    )
    successor = document["execution"]["frontier_refresh_successor"]
    successor.update(
        {
            "new_provider_calls": PRIMARY_TASKS + REPEAT_TASKS,
            "fable_selected_provider_tag": SELECTED_TAG,
            "fable_selected_provider_name": SELECTED_PROVIDER,
            "v42_fable_responses_used_as_score_data": False,
            "v43_complete_fable_block_required": True,
            "full_fable_block_replacement": True,
            "selective_failed_cell_retry": False,
            "aggregate_result_inspected_before_transport_change": True,
            "task_level_scores_or_selections_used_to_select_provider": False,
        }
    )
    document["execution"]["reasoning_control"] = (
        "all v42 prompts, tasks, decoding, output ceiling, reasoning, and concurrency retained; "
        "only Fable's exact OpenRouter upstream changes from Google global to Anthropic"
    )
    prior = int(predecessor["budget"]["prior_frontier_program_spend_micros"])
    v42_spend = int(transport["settled_spend_micros"])
    used = prior + v42_spend
    remaining = PROGRAM_CAP_MICROS - used
    if remaining <= 0:
        raise SelectionPoweredPlanV43Error("frontier refresh exhausted its aggregate cap")
    pricing = candidate.endpoint.get("pricing", {})
    forecast_micros = int(
        (
            Decimal(str(pricing.get("prompt", "0"))) * 4096
            + Decimal(str(pricing.get("completion", "0"))) * 2048
        )
        .scaleb(6)
        .to_integral_value(rounding=ROUND_CEILING)
    )
    if remaining < forecast_micros * (PRIMARY_TASKS + REPEAT_TASKS):
        raise SelectionPoweredPlanV43Error("remaining cap cannot cover the exact request envelope")
    document["budget"]["prior_v42_fable_spend_micros"] = v42_spend
    document["budget"]["prior_frontier_program_spend_micros"] = used
    document["budget"]["hard_cap"] = f"{remaining / 1_000_000:.6f}"
    document["budget"]["successor_scope"] = (
        "one complete 640-primary plus 64-repeat OpenRouter-Anthropic Fable block"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV43Error("constructed v43 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        fable = next(row for row in roster if row["model_id"] == FABLE_ID)
        transport = document["inputs"]["complete_v42_fable_transport"]
        release = document["inputs"]["superseded_v42_release"]
        successor = document["execution"]["frontier_refresh_successor"]
        policy_document = document["execution"]["execution_policy"]
        prior = int(document["budget"]["prior_frontier_program_spend_micros"])
        v42_spend = int(document["budget"]["prior_v42_fable_spend_micros"])
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    policy = selection_execution_policy_v31()
    before = prior - v42_spend
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and fable.get("canonical_model_slug") == "anthropic/claude-5-fable-20260609"
        and fable.get("provider_tag") == SELECTED_TAG
        and fable.get("provider_name") == SELECTED_PROVIDER
        and fable.get("final_max_output_tokens") == 2_048
        and fable.get("final_reasoning_effort") == "minimal"
        and transport.get("primary_status_counts") == {"completed": 413, "failed": 227}
        and transport.get("repeat_status_counts") == {"completed": 43, "failed": 21}
        and transport.get("provider_finish_reason_counts") == {"content_filter": 248, "stop": 456}
        and transport.get("settled_spend_micros") == v42_spend == 2_911_055
        and transport.get("aggregate_score_inspected_before_repair") is True
        and transport.get("complete_old_block_used_as_final_score_data") is False
        and release.get("fable_eligible") is False
        and release.get("fable_completed_primary") == 413
        and release.get("used_as_v43_score_data") is False
        and successor.get("rerun_model_ids") == [FABLE_ID]
        and successor.get("retained_v39_new_model_ids") == [DEEPSEEK_ID]
        and successor.get("fable_selected_provider_tag") == SELECTED_TAG
        and successor.get("v42_fable_responses_used_as_score_data") is False
        and successor.get("v43_complete_fable_block_required") is True
        and successor.get("full_fable_block_replacement") is True
        and successor.get("selective_failed_cell_retry") is False
        and successor.get("aggregate_result_inspected_before_transport_change") is True
        and successor.get("task_level_scores_or_selections_used_to_select_provider") is False
        and successor.get("new_provider_calls") == PRIMARY_TASKS + REPEAT_TASKS
        and prior == 48_026_640
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("hard_cap")
        == f"{(PROGRAM_CAP_MICROS - before - v42_spend) / 1_000_000:.6f}"
        and document["budget"].get("aggregate_program_cap") == "200"
        and document["budget"].get("successor_scope")
        == "one complete 640-primary plus 64-repeat OpenRouter-Anthropic Fable block"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV43Error("content-addressed plan conflict")
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
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--v42-run-directory", type=Path, required=True)
    parser.add_argument("--v42-release", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    transport = _v42_transport_commitment(args.v42_run_directory, plan=predecessor)
    release = _release_commitment(args.v42_release)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        transport=transport,
        release=release,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
