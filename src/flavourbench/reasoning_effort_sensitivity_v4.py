"""Freeze and verify the fail-closed reasoning-effort sensitivity v4.

This module is deliberately split into two evidence layers:

* a six-pair route gate establishes default/high request semantics on every
  exact provider route used by the study; and
* a 24-block, eight-task sensitivity compares the already collected explicit
  low condition with fresh provider-default and explicit-high conditions.

Planning, freezing, verification, and dry runs make no provider or MCP calls.
Paid execution remains impossible until the source-reconstructing route gate,
budget, identity, and Epicure runtime predicates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .frontier_contract_runner import IntegrityError, _verify_live_artifact
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256
from .real_dataset_runner import (
    build_balanced_work_items,
    load_development_task_inventory,
    select_balanced_tasks,
    select_candidates,
)
from .reasoning_effort_route_recovery import verify_v3_route_closure
from .reasoning_effort_sensitivity import (
    _build_runner_task_dossier,
    _derive_runner_manifest,
    _execution_policy_from_base,
    _runner_command,
    _write_manifest,
)
from .response_envelope_route_v4 import (
    _attempt_slots,
    verify_v4_route_acceptance_paths,
)
from .run_journal import JournalIntegrityError, verify_journal_descriptor

HISTORY_SCHEMA = "flavourbench-reasoning-effort-history-audit-v4"
BASELINE_SCHEMA = "flavourbench-reasoning-effort-low-baseline-audit-v4"
ROUTE_PLAN_SCHEMA = "flavourbench-reasoning-effort-route-gate-plan-v4"
STUDY_PLAN_SCHEMA = "flavourbench-reasoning-effort-sensitivity-plan-v4"
RUNNER_ASSETS_SCHEMA = "flavourbench-reasoning-effort-runner-assets-v4"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-preflight-v4"

ROUTE_CONFIRMATION = "RUN_EXACT_REASONING_EFFORT_V4_ROUTE_GATE_6_PAIRS"
FULL_CONFIRMATION = "RUN_EXACT_REASONING_EFFORT_V4_48_NEW_PAIRS"
FREEZE_NONCE = "effort-v4-2026-08-03-source-reconstructed"
NAMESPACE = uuid.UUID("5a046410-9455-42e7-a0fd-f020ce2ade8f")

MODELS = (
    "anthropic/claude-sonnet-5",
    "google/gemini-3.6-flash",
    "deepseek/deepseek-v4-flash-0731",
)
VARIANTS = (
    {
        "variant_id": "provider_default",
        "intermediate_reasoning_effort": None,
        "final_reasoning_effort": None,
        "request_semantics": "reasoning_parameter_omitted",
    },
    {
        "variant_id": "explicit_high",
        "intermediate_reasoning_effort": "high",
        "final_reasoning_effort": "high",
        "request_semantics": "reasoning_effort_explicit_high",
    },
)
TASKS_BY_PANEL = {
    "high_resource_v29": (
        "fb-s0-substitution-003",
        "fb-s0-composition-004",
        "fb-s0-cookability-001",
        "fb-s0-evidence-003",
    ),
    "eight_pair_v27": (
        "fb-s0-substitution-011",
        "fb-s0-composition-003",
        "fb-s0-cookability-015",
        "fb-s0-evidence-004",
    ),
}
PANEL_POLICY_SHA256 = {
    "high_resource_v29": "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d",
    "eight_pair_v27": "5c02a464cf7c6632ea35f88ca9bbf10a527a976d2ceb3a0bd1d62a40cbe1e6c4",
}
EXPECTED_MISSING_LOW = {
    (
        "eight_pair_v27",
        "google/gemini-3.6-flash",
        "fb-s0-composition-003",
    )
}
SOURCE_FILES = (
    "flavourbench/src/flavourbench/provider.py",
    "flavourbench/src/flavourbench/live_smoke.py",
    "flavourbench/src/flavourbench/real_dataset_runner.py",
    "flavourbench/src/flavourbench/reasoning_effort_sensitivity_v4.py",
    "flavourbench/src/flavourbench/run_journal.py",
    "flavourbench/src/flavourbench/execution_policy.py",
    "flavourbench/src/flavourbench/mcp_client.py",
    "flavourbench/requirements.lock",
    "flavourbench/Dockerfile",
)


class SensitivityV4Error(RuntimeError):
    """A frozen input or fail-closed acceptance predicate did not verify."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SensitivityV4Error(f"source must be a regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SensitivityV4Error(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise SensitivityV4Error(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SensitivityV4Error(f"input must be a regular non-symlink file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SensitivityV4Error(f"invalid JSON input: {path}") from error
    if not isinstance(document, dict):
        raise SensitivityV4Error(f"expected a JSON object: {path}")
    return document


def _artifact_verifies(document: object, schema: str) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != schema:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return _is_sha256(digest) and _sha256(unhashed) == digest


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise SensitivityV4Error(f"content-addressed artifact conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _repo_path(repo_root: Path, relative: str | Path) -> Path:
    root = repo_root.resolve()
    path = (root / Path(relative)).resolve()
    if path != root and root not in path.parents:
        raise SensitivityV4Error(f"path escapes repository: {relative}")
    return path


def _relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _source_bundle(repo_root: Path) -> dict[str, Any]:
    files = []
    for relative in SOURCE_FILES:
        path = _repo_path(repo_root, relative)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    return {"files": files, "bundle_sha256": _sha256(files)}


def _content_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    document = _regular_json(path)
    artifact = document.get("artifact_sha256")
    address = document.get("content_address")
    if _is_sha256(artifact):
        unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
        # Historical live-smoke documents used ensure_ascii=True.
        utf8 = _sha256(unhashed)
        ascii_digest = hashlib.sha256(
            json.dumps(unhashed, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        if artifact not in {utf8, ascii_digest}:
            raise SensitivityV4Error(f"artifact content address failed: {path}")
        semantic = str(artifact)
    elif isinstance(address, Mapping) and _is_sha256(address.get("digest")):
        unhashed = {key: value for key, value in document.items() if key != "content_address"}
        semantic = str(address["digest"])
        if (
            address.get("algorithm") != "sha256"
            or address.get("uri") != f"sha256:{semantic}"
            or _sha256(unhashed) != semantic
        ):
            raise SensitivityV4Error(f"manifest content address failed: {path}")
    else:
        semantic = _file_sha256(path)
    return {
        "path": _relative(repo_root, path),
        "semantic_sha256": semantic,
        "file_sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _paths(repo_root: Path) -> dict[str, Path]:
    current = repo_root / "flavourbench/artifacts/season1/current-quality-run"
    return {
        "v1_plan": current
        / "reasoning-effort-sensitivity-v1/reasoning-effort-sensitivity-plan-4fddf823b552f930d807e02c0e9bfb706c5a15020b93244cf9e82d541a177097.json",
        "v1_audit": current
        / "reasoning-effort-sensitivity-v1/reasoning-effort-smoke-audit-da645b267bb52e6eab248d608be181eba0fc14548a0a7d64d952a3091f6ce840.json",
        "v2_plan": current
        / "reasoning-effort-sensitivity-v2-route-validation/reasoning-effort-v2-route-validation-plan-65b64747cbbecf116e3756f69bdbc7c0ccaf1a99a446d3352faa79b432e14e0f.json",
        "v2_audit": current
        / "reasoning-effort-sensitivity-v2-route-validation/final-65b64747/audits/reasoning-effort-v2-route-validation-audit-481303eefacc872701d6a09aa9baeefe887027f655ccdc114ab881c8a16ff821.json",
        "v3_plan": current
        / "reasoning-effort-sensitivity-v3-route-validation/reasoning-effort-v3-route-validation-plan-be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json",
        "v3_audit": current
        / "reasoning-effort-sensitivity-v3-route-validation/final-be2f9d19/audits/reasoning-effort-v3-route-validation-audit-aa66b52d784d813251f7506bbff3eff287f6a94c206fe0550b081ad34a37fb78.json",
        "v3_closure": current
        / "reasoning-effort-sensitivity-v3-route-validation/reasoning-effort-v3-route-closure-290713a8758e9dcabd8567ed086425390537a121385a7ed6c956845d8d3ca1fb.json",
        "v4_plan": current
        / "response-envelope-route-v4/response-envelope-route-v4-plan-a3ef7434064415c93ab78fe818339e0466b100bee01e10e67cbdf1e4d848a4d6.json",
        "v4_audit": current
        / "response-envelope-route-v4/response-envelope-route-v4-audit-70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3.json",
        "v4_closure": current
        / "response-envelope-route-v4/response-envelope-route-v4-closure-dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c.json",
        "manifest_v29": next((current / "manifest-v29-high-resource").glob("*.json")),
        "manifest_v30_generation": next(
            (current / "manifest-v30-floor-replenishment").glob("*.json")
        ),
        "manifest_v27": next((current / "manifest-v27-eight-pairs").glob("*.json")),
        "task_validity": repo_root
        / "flavourbench/artifacts/season1/task-validity/development-v2/development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json",
        "quarantine": current
        / "task-quarantine-v1/current-frontier-task-quarantine-e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15.json",
        "coverage_budget": current
        / "frontier-coverage-repair-execution-v1/frontier-coverage-execution-plan-3f4d1a8135232bb4097b64b6a4c8dae17b20faeb97a65ca5e824a5d3163e5fae.json",
        "epicure_attestation": repo_root
        / "paper/flavourbench/provenance/epicure-runtime-provenance-attestation.json",
    }


def _historical_sources(repo_root: Path) -> list[tuple[str, str, Path]]:
    current = repo_root / "flavourbench/artifacts/season1/current-quality-run"
    records: list[tuple[str, str, Path]] = []
    for variant in ("explicit_low", "provider_default", "explicit_high"):
        root = current / f"reasoning-effort-sensitivity-v1/runs/{variant}/source"
        sources = sorted(root.glob("*.json"))
        if len(sources) != 1:
            raise SensitivityV4Error(f"v1 {variant} must have exactly one immutable source")
        records.append(("v1", variant, sources[0]))
    records.extend(
        [
            (
                "v2",
                "explicit_low",
                current
                / "reasoning-effort-sensitivity-v2-route-validation/final-65b64747/runs/explicit_low/source/20260803T145812Z-1929f13df024.json",
            ),
            (
                "v3",
                "explicit_low",
                current
                / "reasoning-effort-sensitivity-v3-route-validation/final-be2f9d19/runs/explicit_low/source/20260803T152404Z-cb4e3e3dbf36.json",
            ),
        ]
    )
    return records


def build_history_audit(repo_root: Path) -> dict[str, Any]:
    """Re-open the v1-v3 sources and preserve why none estimates an effect."""

    paths = _paths(repo_root)
    source_records: list[dict[str, Any]] = []
    total_provider_requests = 0
    accepted_pairs = 0
    for revision, variant, path in _historical_sources(repo_root):
        try:
            source, digest = _verify_live_artifact(path)
        except IntegrityError as error:
            raise SensitivityV4Error(f"historical source failed verification: {path}") from error
        descriptor = source.get("run_journal")
        if not isinstance(descriptor, Mapping):
            raise SensitivityV4Error(f"historical source has no journal: {path}")
        try:
            entries = verify_journal_descriptor(path.parent, descriptor)
        except JournalIntegrityError as error:
            raise SensitivityV4Error(f"historical journal failed: {path}") from error
        requests = sum(
            entry.get("event_type") == "provider_attempt"
            and isinstance(entry.get("payload"), Mapping)
            and entry["payload"].get("event_type") == "request_started"
            for entry in entries
        )
        total_provider_requests += requests
        complete = source.get("status") == "complete" and set(source.get("results") or {}) == {
            "epicure_off",
            "epicure_on",
        }
        accepted_pairs += int(complete)
        source_records.append(
            {
                "revision": revision,
                "variant_id": variant,
                "source": _content_ref(repo_root, path),
                "artifact_sha256": digest,
                "journal_sha256": descriptor.get("sha256"),
                "provider_requests": requests,
                "status": source.get("status"),
                "usable_pair": complete,
                "raw_provider_body_retained": False,
            }
        )
    v3_plan = _regular_json(paths["v3_plan"])
    v3_audit = _regular_json(paths["v3_audit"])
    v3_closure = _regular_json(paths["v3_closure"])
    if not verify_v3_route_closure(v3_closure):
        raise SensitivityV4Error("v3 permanent closure does not verify")
    if v3_closure.get("v3_route_plan_sha256") != v3_plan.get("artifact_sha256"):
        raise SensitivityV4Error("v3 closure does not bind the selected plan")
    if v3_closure.get("corrected_audit_sha256") != v3_audit.get("artifact_sha256"):
        raise SensitivityV4Error("v3 closure does not bind the selected audit")
    if not verify_v4_route_acceptance_paths(
        plan_path=paths["v4_plan"],
        audit_path=paths["v4_audit"],
        closure_path=paths["v4_closure"],
        repo_root=repo_root,
    ):
        raise SensitivityV4Error("source-reconstructed v4 low route evidence does not verify")
    return {
        "schema_version": HISTORY_SCHEMA,
        "record_role": "source_reconstructed_superseding_history_of_v1_v3_failures",
        "source_records": source_records,
        "source_artifacts": {
            key: _content_ref(repo_root, paths[key])
            for key in (
                "v1_plan",
                "v1_audit",
                "v2_plan",
                "v2_audit",
                "v3_plan",
                "v3_audit",
                "v3_closure",
                "v4_plan",
                "v4_audit",
                "v4_closure",
            )
        },
        "findings": [
            {
                "revision": "v1",
                "finding": "three paid effort pairs failed; legacy HTTP-200 no-choice envelopes cannot be reclassified because raw upstream bodies were intentionally not retained",
            },
            {
                "revision": "v2",
                "finding": "the first pair produced a non-chat envelope and incomplete generation accounting; the remaining two variants were not called",
            },
            {
                "revision": "v3",
                "finding": "the frozen OpenAI route returned four safe HTTP-429 terminal/retry rejections and no accepted generation; remaining variants were not called",
            },
            {
                "revision": "response_envelope_v4",
                "finding": "one low-effort DeepSeek pair passed source reconstruction; it qualifies only the low request path, not omitted-default or explicit-high semantics on all study endpoints",
            },
        ],
        "counts": {
            "historical_source_pairs": len(source_records),
            "historical_usable_pairs": accepted_pairs,
            "historical_provider_requests": total_provider_requests,
            "historical_quality_observations": 0,
            "synthetic_arms": 0,
        },
        "decision": "no_v1_v3_record_can_estimate_reasoning_effort_sensitivity",
        "provider_calls_made_by_audit": False,
        "epicure_calls_made_by_audit": False,
        "claim_boundary": {
            "quality_effect_estimable": False,
            "rank_eligible": False,
            "official": False,
        },
    }


def _manifest(path: Path) -> dict[str, Any]:
    document = _regular_json(path)
    if not verify_manifest_content_address(document):
        raise SensitivityV4Error(f"manifest content address failed: {path}")
    return document


def _model_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("models")
    if not isinstance(records, list):
        raise SensitivityV4Error("manifest has no model records")
    result = {
        str((record.get("model") or {}).get("id") or ""): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("model"), Mapping)
    }
    if not set(MODELS).issubset(result):
        raise SensitivityV4Error("manifest lacks a selected sensitivity endpoint")
    return result


def _task_map(path: Path) -> dict[str, Mapping[str, Any]]:
    document = _regular_json(path)
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise SensitivityV4Error("task-validity artifact has no tasks")
    return {
        str(task.get("task_id") or ""): task
        for task in tasks
        if isinstance(task, Mapping) and task.get("task_id")
    }


def _all_source_documents(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    current = repo_root / "flavourbench/artifacts/season1/current-quality-run"
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(current.glob("**/source/*.json")):
        try:
            document = _regular_json(path)
        except SensitivityV4Error:
            continue
        if document.get("schema_version") == "flavourbench-live-smoke-v1":
            records.append((path, document))
    return records


def _complete_source_predicates(
    *,
    source: Mapping[str, Any],
    model_record: Mapping[str, Any],
    task: Mapping[str, Any],
    policy_sha256: str,
    expected_manifest_sha256: str,
) -> list[str]:
    failures: list[str] = []
    model = model_record.get("model") or {}
    endpoint = model_record.get("endpoint") or {}
    if source.get("status") != "complete" or source.get("errors") != {}:
        failures.append("source_not_complete")
    if source.get("execution_policy_sha256") != policy_sha256:
        failures.append("execution_policy_mismatch")
    reasoning = (source.get("execution_policy") or {}).get("reasoning") or {}
    if reasoning.get("intermediate_effort") != "low" or reasoning.get("final_effort") != "low":
        failures.append("not_explicit_low")
    if (
        source.get("requested_model_id") != model.get("id")
        or source.get("requested_provider") != endpoint.get("tag")
        or source.get("candidate_manifest_sha256") != expected_manifest_sha256
        or source.get("dataset_task_id") != task.get("task_id")
        or source.get("prompt_sha256") != task.get("prompt_sha256")
    ):
        failures.append("frozen_coordinate_mismatch")
    results = source.get("results")
    if not isinstance(results, Mapping) or set(results) != {"epicure_off", "epicure_on"}:
        failures.append("condition_pair_incomplete")
        return failures
    for condition in ("epicure_off", "epicure_on"):
        arm = results.get(condition)
        if not isinstance(arm, Mapping):
            failures.append(f"{condition}_arm_missing")
            continue
        if (
            arm.get("actual_model_id") != model.get("canonical_slug")
            or arm.get("actual_provider") != endpoint.get("provider_name")
            or arm.get("finish_reason") != "stop"
            or arm.get("cost_reconciled") is not True
            or len(str(arm.get("answer_markdown") or "").strip()) < 100
        ):
            failures.append(f"{condition}_identity_finish_cost_or_content")
        traces = arm.get("tool_trace")
        if not isinstance(traces, list):
            failures.append(f"{condition}_tool_trace_missing")
        elif condition == "epicure_off" and traces:
            failures.append("epicure_off_tool_leakage")
        elif condition == "epicure_on" and not any(
            isinstance(trace, Mapping) and trace.get("is_error") is False for trace in traces
        ):
            failures.append("epicure_on_no_successful_tool_call")
    return failures


def build_low_baseline_audit(repo_root: Path) -> dict[str, Any]:
    """Reconstruct 24 low cells from their raw source artifacts and journals."""

    paths = _paths(repo_root)
    manifests = {
        "high_resource_v29": _manifest(paths["manifest_v29"]),
        "eight_pair_v27": _manifest(paths["manifest_v27"]),
    }
    generation_manifests = {
        "high_resource_v29": {
            "floor_replenishment": _manifest(paths["manifest_v30_generation"]),
            "coverage_repair": manifests["high_resource_v29"],
        },
        "eight_pair_v27": {"eight_pair": manifests["eight_pair_v27"]},
    }
    # v30 changed only per-model cost forecasts from v29. Holding every selected
    # endpoint field fixed is necessary before their low responses can be pooled.
    for model_id in MODELS:
        current_record = _model_map(manifests["high_resource_v29"])[model_id]
        generation_record = _model_map(
            generation_manifests["high_resource_v29"]["floor_replenishment"]
        )[model_id]
        if {key: value for key, value in current_record.items() if key != "forecast"} != {
            key: value for key, value in generation_record.items() if key != "forecast"
        }:
            raise SensitivityV4Error(f"v29/v30 selected endpoint contract drifted for {model_id}")
    task_records = _task_map(paths["task_validity"])
    quarantine = _regular_json(paths["quarantine"])
    quarantined = {str(record["task_id"]) for record in quarantine.get("records") or []}
    selected_tasks = {task for tasks in TASKS_BY_PANEL.values() for task in tasks}
    if selected_tasks & quarantined:
        raise SensitivityV4Error("sensitivity task set intersects the current quarantine")
    sources = _all_source_documents(repo_root)
    cells: list[dict[str, Any]] = []
    complete = 0
    missing = 0
    for panel, task_ids in TASKS_BY_PANEL.items():
        model_records = _model_map(manifests[panel])
        policy_sha = PANEL_POLICY_SHA256[panel]
        for model_id in MODELS:
            model_record = model_records[model_id]
            for task_id in task_ids:
                task = task_records.get(task_id)
                if not isinstance(task, Mapping):
                    raise SensitivityV4Error(f"selected task is absent: {task_id}")
                if panel == "high_resource_v29" and task_id != "fb-s0-cookability-001":
                    generation_manifest = generation_manifests[panel]["floor_replenishment"]
                elif panel == "high_resource_v29":
                    generation_manifest = generation_manifests[panel]["coverage_repair"]
                else:
                    generation_manifest = generation_manifests[panel]["eight_pair"]
                expected_manifest_sha256 = str(generation_manifest["content_address"]["digest"])
                candidates = [
                    (path, document)
                    for path, document in sources
                    if document.get("requested_model_id") == model_id
                    and document.get("dataset_task_id") == task_id
                    and document.get("execution_policy_sha256") == policy_sha
                ]
                good: list[tuple[Path, dict[str, Any]]] = []
                for path, _source in candidates:
                    try:
                        verified, _ = _verify_live_artifact(path)
                    except IntegrityError:
                        continue
                    failures = _complete_source_predicates(
                        source=verified,
                        model_record=model_record,
                        task=task,
                        policy_sha256=policy_sha,
                        expected_manifest_sha256=expected_manifest_sha256,
                    )
                    if not failures:
                        good.append((path, verified))
                coordinate = (panel, model_id, task_id)
                if coordinate in EXPECTED_MISSING_LOW:
                    if good:
                        raise SensitivityV4Error(
                            "prespecified missing low cell unexpectedly passed"
                        )
                    failed = [
                        (path, source)
                        for path, source in candidates
                        if source.get("status") == "failed_or_unreconciled"
                    ]
                    if len(failed) != 1:
                        raise SensitivityV4Error("missing low cell lacks one immutable failure")
                    path, source = failed[0]
                    try:
                        verified, digest = _verify_live_artifact(path)
                        descriptor = verified.get("run_journal")
                        if not isinstance(descriptor, Mapping):
                            raise SensitivityV4Error("failed low source has no journal")
                        verify_journal_descriptor(path.parent, descriptor)
                    except (IntegrityError, JournalIntegrityError) as error:
                        raise SensitivityV4Error(
                            "failed low source does not reconstruct"
                        ) from error
                    cells.append(
                        {
                            "panel_id": panel,
                            "model_id": model_id,
                            "task_id": task_id,
                            "task_family": task.get("family"),
                            "execution_policy_sha256": policy_sha,
                            "candidate_manifest_sha256": expected_manifest_sha256,
                            "status": "immutable_missing_low_due_prior_failure",
                            "source": _content_ref(repo_root, path),
                            "source_artifact_sha256": digest,
                            "journal_sha256": descriptor.get("sha256"),
                            "retry_authorized": False,
                            "quality_contrasts_requiring_low": False,
                        }
                    )
                    missing += 1
                    continue
                if len(good) != 1:
                    raise SensitivityV4Error(
                        f"expected one source-reconstructable low pair for {coordinate}; got {len(good)}"
                    )
                path, source = good[0]
                descriptor = source.get("run_journal")
                if not isinstance(descriptor, Mapping):
                    raise SensitivityV4Error(f"low source has no journal: {path}")
                try:
                    verify_journal_descriptor(path.parent, descriptor)
                except JournalIntegrityError as error:
                    raise SensitivityV4Error(f"low source journal failed: {path}") from error
                cells.append(
                    {
                        "panel_id": panel,
                        "model_id": model_id,
                        "canonical_model_slug": (model_record["model"])["canonical_slug"],
                        "provider_endpoint": (model_record["endpoint"])["tag"],
                        "task_id": task_id,
                        "task_family": task.get("family"),
                        "prompt_sha256": task.get("prompt_sha256"),
                        "execution_policy_sha256": policy_sha,
                        "candidate_manifest_sha256": expected_manifest_sha256,
                        "status": "source_reconstructed_complete_low_pair",
                        "source": _content_ref(repo_root, path),
                        "source_artifact_sha256": source.get("artifact_sha256"),
                        "journal_sha256": descriptor.get("sha256"),
                        "quality_contrasts_requiring_low": True,
                        "request_routing_reconstruction": {
                            "raw_request_body_retained": False,
                            "provider_controls_independently_reconstructed": False,
                            "actual_model_and_provider_verified": True,
                            "limitation": "historical journals retain request-payload hashes, not raw request bodies",
                        },
                    }
                )
                complete += 1
    family_counts = Counter(cell["task_family"] for cell in cells)
    if (
        len(cells) != 24
        or complete != 23
        or missing != 1
        or family_counts
        != Counter(
            {family: 6 for family in ("substitution", "composition", "cookability", "evidence")}
        )
    ):
        raise SensitivityV4Error("low baseline balance or completeness invariant failed")
    cells.sort(key=lambda item: (item["panel_id"], item["model_id"], item["task_id"]))
    return {
        "schema_version": BASELINE_SCHEMA,
        "record_role": "source_reconstructed_explicit_low_baseline_for_effort_v4",
        "source_artifacts": {
            "manifest_v29": _content_ref(repo_root, paths["manifest_v29"]),
            "manifest_v30_generation": _content_ref(repo_root, paths["manifest_v30_generation"]),
            "manifest_v27": _content_ref(repo_root, paths["manifest_v27"]),
            "task_validity": _content_ref(repo_root, paths["task_validity"]),
            "quarantine": _content_ref(repo_root, paths["quarantine"]),
        },
        "selection": {
            "models": list(MODELS),
            "tasks_by_panel": {key: list(value) for key, value in TASKS_BY_PANEL.items()},
            "quality_outcomes_used_to_choose_tasks_or_models": 0,
            "quarantined_tasks": 0,
            "synthetic_tasks": 0,
            "rule": "two previously frozen one-per-family panels; no replacement based on answer quality",
        },
        "cells": cells,
        "counts": {
            "model_task_blocks": 24,
            "source_reconstructed_complete_low_pairs": 23,
            "immutable_missing_low_pairs": 1,
            "usable_low_arms": 46,
            "synthetic_arms": 0,
            "per_family_blocks": dict(sorted(family_counts.items())),
        },
        "missingness_policy": {
            "retry_to_complete_prohibited": True,
            "failed_pair_retained_in_reliability": True,
            "new_default_vs_high_contrast_for_cell_allowed": True,
            "contrasts_requiring_low_for_cell_excluded": True,
        },
        "decision": "low_baseline_verified_with_one_prespecified_missing_cell",
        "provider_calls_made_by_audit": False,
        "epicure_calls_made_by_audit": False,
        "claim_boundary": {"official": False, "rank_eligible": False},
    }


def _selected_model_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = _model_map(manifest)
    records: list[dict[str, Any]] = []
    for model_id in MODELS:
        record = by_id[model_id]
        model = record["model"]
        endpoint = record["endpoint"]
        reasoning = model.get("reasoning") or {}
        supported = sorted(str(value) for value in reasoning.get("supported_efforts") or [])
        if not {"low", "high"}.issubset(supported):
            raise SensitivityV4Error(f"route lacks low/high support: {model_id}")
        records.append(
            {
                "model_id": model_id,
                "canonical_model_slug": model.get("canonical_slug"),
                "provider_endpoint": endpoint.get("tag"),
                "actual_provider_name": endpoint.get("provider_name"),
                "endpoint_document_sha256": record.get("endpoint_document_sha256"),
                "endpoint_execution_contract_sha256": endpoint_execution_contract_sha256(endpoint),
                "provider_controls": (record.get("request_policy") or {}).get("provider"),
                "execution_backend": (record.get("execution_route") or {}).get("selected_backend"),
                "supported_efforts": supported,
                "provider_default_effort": reasoning.get("default_effort"),
                "provider_default_mandatory": reasoning.get("mandatory"),
                "four_pair_worst_case_usd": str(
                    (record.get("forecast") or {}).get("model_block_worst_case_usd")
                ),
            }
        )
    return records


def _route_work_items(
    *,
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    epicure: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Decimal]:
    model_records = _model_map(manifest)
    work: list[dict[str, Any]] = []
    total = Decimal(0)
    for model_id in MODELS:
        record = model_records[model_id]
        per_pair = _decimal(
            (record.get("forecast") or {}).get("model_block_worst_case_usd"),
            field=f"{model_id} four-pair forecast",
        ) / Decimal(4)
        for variant in VARIANTS:
            route_coordinate = {
                "schema_version": "flavourbench-reasoning-effort-route-coordinate-v4",
                "freeze_nonce": FREEZE_NONCE,
                "model_id": model_id,
                "canonical_model_slug": (record["model"])["canonical_slug"],
                "provider_endpoint": (record["endpoint"])["tag"],
                "actual_provider_name": (record["endpoint"])["provider_name"],
                "endpoint_execution_contract_sha256": endpoint_execution_contract_sha256(
                    record["endpoint"]
                ),
                "provider_controls": (record.get("request_policy") or {}).get("provider"),
                "task_id": task["task_id"],
                "prompt_sha256": task["prompt_sha256"],
                "variant_id": variant["variant_id"],
                "intermediate_reasoning_effort": variant["intermediate_reasoning_effort"],
                "final_reasoning_effort": variant["final_reasoning_effort"],
                "epicure_bundle_sha256": epicure.get("bundle_sha256"),
                "epicure_application_sha256": epicure.get("application_sha256"),
                "epicure_tool_schema_sha256": epicure.get("tool_schema_sha256"),
            }
            route_cell_id = _sha256(route_coordinate)
            work_item_id = _sha256({"route_cell_id": route_cell_id, "role": "effort-v4-gate"})
            run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
            work.append(
                {
                    "route_cell_id": route_cell_id,
                    "work_item_id": work_item_id,
                    "run_id": run_id,
                    "arm_ids": [f"{run_id}:epicure_off", f"{run_id}:epicure_on"],
                    "attempt_slots": _attempt_slots(run_id, route_cell_id, FREEZE_NONCE),
                    "route_coordinate": route_coordinate,
                    "worst_case_reserve_usd": _decimal_text(per_pair),
                    "diagnostic_outputs_reused": False,
                }
            )
            total += per_pair
    # Cheapest route first; within route, default before high. A failure closes the suffix.
    model_order = {
        "deepseek/deepseek-v4-flash-0731": 0,
        "google/gemini-3.6-flash": 1,
        "anthropic/claude-sonnet-5": 2,
    }
    variant_order = {"provider_default": 0, "explicit_high": 1}
    work.sort(
        key=lambda item: (
            model_order[item["route_coordinate"]["model_id"]],
            variant_order[item["route_coordinate"]["variant_id"]],
        )
    )
    return work, total


def _epicure_contract(attestation: Mapping[str, Any]) -> dict[str, Any]:
    # The public attestation uses a small, stable projection consumed elsewhere in the paper.
    runtime = attestation.get("runtime") or attestation
    return {
        "release_id": runtime.get("release_id")
        or attestation.get("epicure_release_id")
        or "exploratory-unmatched-1790-runtime",
        "bundle_sha256": runtime.get("bundle_sha256")
        or attestation.get("bundle_sha256")
        or "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1",
        "application_sha256": runtime.get("application_sha256")
        or attestation.get("application_sha256")
        or "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313",
        "tool_schema_sha256": runtime.get("tool_schema_sha256")
        or attestation.get("tool_schema_sha256")
        or "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd",
        "public_reconstruction_complete": False,
        "rank_eligible": False,
    }


def build_route_plan(
    repo_root: Path,
    *,
    history_sha256: str,
    baseline_sha256: str,
) -> dict[str, Any]:
    paths = _paths(repo_root)
    manifest = _manifest(paths["manifest_v29"])
    tasks = _task_map(paths["task_validity"])
    task = tasks["fb-s0-substitution-003"]
    epicure = _epicure_contract(_regular_json(paths["epicure_attestation"]))
    work, total = _route_work_items(manifest=manifest, task=task, epicure=epicure)
    coverage = _regular_json(paths["coverage_budget"])
    baseline_projected = _decimal(
        (coverage.get("budget") or {}).get("projected_total_exposure_usd"),
        field="post-coverage projected exposure",
    )
    ceiling = _decimal(
        (coverage.get("budget") or {}).get("admission_ceiling_usd"),
        field="admission ceiling",
    )
    hard = _decimal((coverage.get("budget") or {}).get("hard_cap_usd"), field="hard cap")
    return {
        "schema_version": ROUTE_PLAN_SCHEMA,
        "study_id": "frontier-reasoning-effort-sensitivity-v4",
        "record_role": "six_pair_source_reconstructing_default_high_route_gate",
        "history_audit_sha256": history_sha256,
        "low_baseline_audit_sha256": baseline_sha256,
        "source_artifacts": {
            "manifest_v29": _content_ref(repo_root, paths["manifest_v29"]),
            "task_validity": _content_ref(repo_root, paths["task_validity"]),
            "coverage_budget": _content_ref(repo_root, paths["coverage_budget"]),
            "epicure_attestation": _content_ref(repo_root, paths["epicure_attestation"]),
            "v4_low_plan": _content_ref(repo_root, paths["v4_plan"]),
            "v4_low_audit": _content_ref(repo_root, paths["v4_audit"]),
            "v4_low_closure": _content_ref(repo_root, paths["v4_closure"]),
        },
        "source_code": _source_bundle(repo_root),
        "task": {
            "task_id": task["task_id"],
            "family": task["family"],
            "prompt": task["prompt"],
            "prompt_sha256": task["prompt_sha256"],
            "synthetic": False,
            "quarantined": False,
        },
        "models": _selected_model_records(manifest),
        "variants": list(VARIANTS),
        "epicure": epicure,
        "work_items": work,
        "execution_order": [item["work_item_id"] for item in work],
        "counts": {
            "matched_pairs": 6,
            "response_arms": 12,
            "models": 3,
            "effort_variants": 2,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "acceptance": {
            "all_six_pairs_required": True,
            "each_arm_substantive": True,
            "each_epicure_on_arm_has_successful_real_tool_call": True,
            "each_epicure_off_arm_has_zero_tool_calls": True,
            "request_contract_reconstructed_from_journal": True,
            "provider_default_requires_reasoning_field_absent": True,
            "explicit_high_requires_reasoning_effort_high": True,
            "all_generation_costs_reconciled": True,
            "identity_substitution_allowed": False,
            "diagnostic_outputs_enter_quality_fit": False,
            "stop_and_close_suffix_on_first_failure": True,
            "replay_permitted": False,
        },
        "budget": {
            "currency": "USD",
            "post_coverage_projected_exposure_usd": _decimal_text(baseline_projected),
            "route_gate_worst_case_usd": _decimal_text(total),
            "projected_after_route_gate_usd": _decimal_text(baseline_projected + total),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "hard_cap_usd": _decimal_text(hard),
            "admitted_for_route_gate": baseline_projected + total <= ceiling,
        },
        "execution": {
            "confirmation": ROUTE_CONFIRMATION,
            "provider_calls_made_by_plan": False,
            "epicure_calls_made_by_plan": False,
            "status": "frozen_not_executed",
        },
        "claim_boundary": {
            "route_gate_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_effect_estimable": False,
        },
    }


def build_study_plan(
    repo_root: Path,
    *,
    history_sha256: str,
    baseline: Mapping[str, Any],
    route_plan: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _paths(repo_root)
    manifests = {
        "high_resource_v29": _manifest(paths["manifest_v29"]),
        "eight_pair_v27": _manifest(paths["manifest_v27"]),
    }
    panel_forecasts: dict[str, Decimal] = {}
    panel_models: dict[str, list[dict[str, Any]]] = {}
    for panel, manifest in manifests.items():
        selected = _selected_model_records(manifest)
        panel_models[panel] = selected
        source_pairs = int((manifest.get("run_design") or {}).get("assignments_per_model") or 0)
        if source_pairs not in {4, 8}:
            raise SensitivityV4Error("panel forecast lacks a four/eight-pair basis")
        selected_pairs = len(TASKS_BY_PANEL[panel])
        panel_forecasts[panel] = sum(
            (
                _decimal(model["four_pair_worst_case_usd"], field="model forecast")
                * Decimal(selected_pairs)
                / Decimal(source_pairs)
                for model in selected
            ),
            Decimal(0),
        )
    per_variant = sum(panel_forecasts.values(), Decimal(0))
    study_forecast = per_variant * Decimal(len(VARIANTS))
    route_forecast = _decimal(
        route_plan["budget"]["route_gate_worst_case_usd"], field="route forecast"
    )
    coverage_projected = _decimal(
        route_plan["budget"]["post_coverage_projected_exposure_usd"],
        field="coverage projected",
    )
    ceiling = _decimal(route_plan["budget"]["admission_ceiling_usd"], field="ceiling")
    hard = _decimal(route_plan["budget"]["hard_cap_usd"], field="hard")
    projected = coverage_projected + route_forecast + study_forecast
    task_records = _task_map(paths["task_validity"])
    tasks = [
        {
            "panel_id": panel,
            "task_id": task_id,
            "family": task_records[task_id]["family"],
            "prompt_sha256": task_records[task_id]["prompt_sha256"],
            "execution_policy_sha256": PANEL_POLICY_SHA256[panel],
            "synthetic": False,
            "quarantined": False,
        }
        for panel, task_ids in TASKS_BY_PANEL.items()
        for task_id in task_ids
    ]
    return {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": "frontier-reasoning-effort-sensitivity-v4",
        "record_role": "minimum_family_balanced_fixed_endpoint_reasoning_sensitivity",
        "history_audit_sha256": history_sha256,
        "low_baseline_audit_sha256": baseline["artifact_sha256"],
        "route_gate_plan_sha256": route_plan["artifact_sha256"],
        "source_artifacts": {
            "manifest_v29": _content_ref(repo_root, paths["manifest_v29"]),
            "manifest_v27": _content_ref(repo_root, paths["manifest_v27"]),
            "task_validity": _content_ref(repo_root, paths["task_validity"]),
            "quarantine": _content_ref(repo_root, paths["quarantine"]),
            "coverage_budget": _content_ref(repo_root, paths["coverage_budget"]),
            "epicure_attestation": _content_ref(repo_root, paths["epicure_attestation"]),
        },
        "source_code": _source_bundle(repo_root),
        "models": panel_models["high_resource_v29"],
        "tasks": tasks,
        "panels": [
            {
                "panel_id": panel,
                "base_manifest_sha256": manifest["content_address"]["digest"],
                "base_execution_policy_sha256": PANEL_POLICY_SHA256[panel],
                "tasks": list(TASKS_BY_PANEL[panel]),
                "per_variant_worst_case_usd": _decimal_text(panel_forecasts[panel]),
            }
            for panel, manifest in manifests.items()
        ],
        "reasoning_variants": [
            {
                "variant_id": "explicit_low",
                "source": "source_reconstructed_prior_real_pairs",
                "intermediate_reasoning_effort": "low",
                "final_reasoning_effort": "low",
                "new_provider_calls": 0,
            },
            *[dict(variant) for variant in VARIANTS],
        ],
        "factorial_design": {
            "model_task_blocks": 24,
            "models": 3,
            "tasks": 8,
            "tasks_per_family": 2,
            "conditions": ["epicure_off", "epicure_on"],
            "effort_levels": ["explicit_low", "provider_default", "explicit_high"],
            "complete_low_blocks": 23,
            "prespecified_missing_low_blocks": 1,
            "new_default_high_pairs": 48,
            "new_response_arms": 96,
            "route_gate_pairs_excluded": 6,
            "synthetic_arms": 0,
            "pair_scheduling": "effort_order_counterbalanced_within_panel_ordinal",
        },
        "analysis_contract": {
            "primary_estimands": [
                "provider-default minus explicit-low within model, task, and Epicure condition",
                "explicit-high minus explicit-low within model, task, and Epicure condition",
                "explicit-high minus provider-default within model, task, and Epicure condition",
                "effort-by-Epicure interaction within model and task",
            ],
            "quality_source": "new blinded human judgments only",
            "operational_outcomes": [
                "normal-finish rate",
                "Epicure treatment success",
                "cost",
                "latency",
                "tool-call count",
            ],
            "missingness": (
                "the one pre-existing failed low cell remains in reliability; it is excluded "
                "only from contrasts requiring low and is never retried to obtain a favorable arm"
            ),
            "unit_of_analysis": "model-by-task block",
            "dependence": (
                "arms and repeated presentations are nested in model-by-task; task-cluster and "
                "model-stratified resampling is required"
            ),
            "primary_test": (
                "two-sided exact paired sign/randomization test over 24 fixed blocks for "
                "default-versus-high; low contrasts use 23 complete blocks"
            ),
            "multiplicity": (
                "Holm correction across the three aggregate effort contrasts; family and model "
                "effects are descriptive with intervals"
            ),
            "power_boundary": {
                "blocks": 24,
                "two_sided_null_alpha_at_18_of_24_same_direction": "0.022655844688415527",
                "power_if_true_direction_probability_is_0_8": "0.8110710551188802",
                "interpretation": "powered only for a large, consistent aggregate shift",
            },
            "ranking_use": "prohibited; fixed-endpoint development sensitivity only",
        },
        "budget": {
            "currency": "USD",
            "post_coverage_projected_exposure_usd": _decimal_text(coverage_projected),
            "route_gate_worst_case_usd": _decimal_text(route_forecast),
            "per_variant_study_worst_case_usd": _decimal_text(per_variant),
            "full_study_worst_case_usd": _decimal_text(study_forecast),
            "total_new_worst_case_usd": _decimal_text(route_forecast + study_forecast),
            "projected_total_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "hard_cap_usd": _decimal_text(hard),
            "admitted_after_route_gate_if_budget_unchanged": projected <= ceiling,
            "transactional_rule": (
                "re-run the global source-backed audit before each one-pair admission; stop at "
                "85 percent, drain at 95 percent, and hard-stop at 100 percent"
            ),
        },
        "admission": {
            "decision": "blocked_pending_source_reconstructed_six_pair_route_gate",
            "required_route_gate_pairs": 6,
            "require_current_source_bundle": True,
            "require_no_active_or_orphan_reservation": True,
            "require_fresh_budget_at_or_below_admission_ceiling": True,
            "provider_calls_made_by_plan": False,
            "epicure_calls_made_by_plan": False,
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "enters_primary_leaderboard": False,
            "generalizes_to_unmeasured_models": False,
            "generalizes_to_all_culinary_tasks": False,
        },
    }


def _payload_with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return {**unhashed, "artifact_sha256": _sha256(unhashed)}


def _panel_plan_view(study: Mapping[str, Any], panel: Mapping[str, Any]) -> dict[str, Any]:
    panel_id = str(panel["panel_id"])
    anchors = [task for task in study["tasks"] if task["panel_id"] == panel_id]
    return {
        "artifact_sha256": study["artifact_sha256"],
        "task_design": {
            "anchors": anchors,
            "coverage_schedule_sha256": study["low_baseline_audit_sha256"],
        },
        "model_design": {"models": study["models"]},
        "budget": {
            "per_variant_worst_case_usd": panel["per_variant_worst_case_usd"],
            "hard_cap_usd": study["budget"]["hard_cap_usd"],
            "admission_ceiling_usd": study["budget"]["admission_ceiling_usd"],
        },
        "epicure": _epicure_contract({}),
    }


def _task_dossier_for_panel(
    *,
    study: Mapping[str, Any],
    panel: Mapping[str, Any],
    source_validity: Mapping[str, Any],
) -> dict[str, Any]:
    view = _panel_plan_view(study, panel)
    payload = _build_runner_task_dossier(plan=view, source_task_validity=source_validity)
    payload["selection_policy"] = {
        "method": "reasoning_effort_v4_prespecified_panel",
        "panel_id": panel["panel_id"],
        "study_plan_sha256": study["artifact_sha256"],
        "quality_observations_used": 0,
    }
    return payload


def _materialized_work_items(
    *, manifest: Mapping[str, Any], dossier_path: Path, variant: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tasks, _ = load_development_task_inventory(dossier_path)
    design = manifest["run_design"]
    selected, registry = select_balanced_tasks(
        tasks_per_family=1,
        seed=design["selection_seed"],
        tasks=tasks,
    )
    candidates = select_candidates(manifest, ())
    policy = _execution_policy_from_base(
        manifest,
        intermediate_effort=variant["intermediate_reasoning_effort"],
        final_effort=variant["final_reasoning_effort"],
    )
    items = build_balanced_work_items(
        manifest_sha256=manifest["content_address"]["digest"],
        task_registry_digest=registry,
        selected_tasks=selected,
        candidates=candidates,
        execution_policy=policy,
        assignments_per_model=4,
    )
    return [
        {
            "ordinal": item.ordinal,
            "work_item_id": item.work_item_id,
            "model_id": item.candidate.model_id,
            "task_id": item.task.public_id,
            "task_family": item.task.family,
        }
        for item in items
    ]


def materialize_runner_assets(
    *,
    repo_root: Path,
    study: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if not _artifact_verifies(study, STUDY_PLAN_SCHEMA):
        raise SensitivityV4Error("study plan does not verify")
    paths = _paths(repo_root)
    source_validity = _regular_json(paths["task_validity"])
    base_by_panel = {
        "high_resource_v29": _manifest(paths["manifest_v29"]),
        "eight_pair_v27": _manifest(paths["manifest_v27"]),
    }
    variants: list[dict[str, Any]] = []
    by_panel_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for panel in study["panels"]:
        panel_id = str(panel["panel_id"])
        view = _panel_plan_view(study, panel)
        dossier_payload = _task_dossier_for_panel(
            study=study, panel=panel, source_validity=source_validity
        )
        dossier_path = _write_artifact(
            output_dir / "tasks",
            f"reasoning-effort-v4-{panel_id}-tasks",
            dossier_payload,
        )
        dossier = _regular_json(dossier_path)
        task_inventory, _ = load_development_task_inventory(dossier_path)
        from .real_dataset_runner import task_registry_sha256

        registry = task_registry_sha256(task_inventory)
        for variant in VARIANTS:
            derived = _derive_runner_manifest(
                plan=view,
                base_manifest=base_by_panel[panel_id],
                task_dossier_path=dossier_path,
                task_dossier=dossier,
                task_registry_digest=registry,
                variant=variant,
            )
            manifest_path = _write_manifest(output_dir / "manifests", derived)
            run_root = output_dir / "runs" / panel_id / variant["variant_id"]
            command = _runner_command(
                manifest_path=manifest_path,
                manifest=derived,
                task_dossier_path=dossier_path,
                variant=variant,
                run_root=run_root,
            )
            # Never expose a command that can consume the whole panel in one accidental call.
            command.extend(["--max-new-pairs", "1"])
            record = {
                "panel_id": panel_id,
                "variant_id": variant["variant_id"],
                "manifest": _relative(repo_root, manifest_path),
                "manifest_sha256": derived["content_address"]["digest"],
                "task_dossier": _relative(repo_root, dossier_path),
                "task_dossier_sha256": dossier["artifact_sha256"],
                "run_root": _relative(repo_root, run_root),
                "single_pair_dry_run_command": command,
                "single_pair_live_suffix": [
                    "--execute",
                    "--confirm",
                    "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
                ],
                "work_items": _materialized_work_items(
                    manifest=derived, dossier_path=dossier_path, variant=variant
                ),
            }
            variants.append(record)
            by_panel_variant[(panel_id, variant["variant_id"])] = record
    schedule: list[dict[str, Any]] = []
    for panel in ("high_resource_v29", "eight_pair_v27"):
        defaults = by_panel_variant[(panel, "provider_default")]["work_items"]
        highs = by_panel_variant[(panel, "explicit_high")]["work_items"]
        if len(defaults) != 12 or len(highs) != 12:
            raise SensitivityV4Error("panel variant workload is not 12 matched pairs")
        blocks: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for default, high in zip(defaults, highs, strict=True):
            if (default["model_id"], default["task_id"], default["ordinal"]) != (
                high["model_id"],
                high["task_id"],
                high["ordinal"],
            ):
                raise SensitivityV4Error("effort variants do not share one block order")
            coordinate = {
                "panel_id": panel,
                "model_id": default["model_id"],
                "task_id": default["task_id"],
                "panel_ordinal": default["ordinal"],
            }
            blocks.append((_sha256(coordinate), default, high, coordinate))
        # Exact 1:1 first-period balance within each panel, with deterministic
        # hash ordering fixed before any quality outcome exists.
        blocks.sort(key=lambda block: block[0])
        for index, (_, default, high, coordinate) in enumerate(blocks):
            order = (
                ["provider_default", "explicit_high"]
                if index < len(blocks) // 2
                else ["explicit_high", "provider_default"]
            )
            for variant_id in order:
                item = default if variant_id == "provider_default" else high
                schedule.append(
                    {
                        "invocation_ordinal": len(schedule) + 1,
                        **coordinate,
                        "variant_id": variant_id,
                        "expected_work_item_id": item["work_item_id"],
                        "effort_order_in_block": order,
                    }
                )
    return {
        "schema_version": RUNNER_ASSETS_SCHEMA,
        "record_role": "single_pair_counterbalanced_append_only_runner_schedule",
        "study_plan_sha256": study["artifact_sha256"],
        "variants": variants,
        "execution_schedule": schedule,
        "execution_command": {
            "module": "flavourbench.reasoning_effort_sensitivity_v4",
            "subcommand": "execute-full",
            "confirmation": FULL_CONFIRMATION,
            "implemented": False,
            "reason": (
                "paid execution remains deliberately unavailable until a source-reconstructed "
                "six-pair route closure is supplied"
            ),
        },
        "operator_contract": (
            "After a passing route closure, invoke exactly one scheduled single-pair command at "
            "a time, re-audit budget and source state, and verify its immutable journal before "
            "the next invocation. Never run a manifest without --max-new-pairs 1."
        ),
        "counts": {
            "manifests": 4,
            "new_matched_pairs": 48,
            "new_response_arms": 96,
            "schedule_invocations": 48,
            "synthetic_arms": 0,
        },
        "provider_calls_made_by_materialization": False,
        "epicure_calls_made_by_materialization": False,
        "official": False,
        "rank_eligible": False,
    }


def build_preflight(
    *,
    study: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "no_call_fail_closed_reasoning_effort_v4_preflight",
        "study_plan_sha256": study["artifact_sha256"],
        "route_gate_plan_sha256": route_plan["artifact_sha256"],
        "runner_assets_sha256": runner_assets["artifact_sha256"],
        "checks": {
            "history_source_reconstructed": True,
            "low_baseline_source_reconstructed": True,
            "quarantined_tasks": 0,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
            "all_selected_routes_support_low_and_high": True,
            "projected_budget_at_or_below_admission_ceiling": study["budget"][
                "admitted_after_route_gate_if_budget_unchanged"
            ],
            "route_gate_pass_receipt_present": False,
            "human_quality_observations": 0,
        },
        "decision": "blocked_before_full_provider_calls_pending_six_pair_route_gate",
        "only_authorized_next_external_action": (
            "execute the exact six-pair route gate after re-verifying environment, source, "
            "Epicure identity, and a fresh global budget audit"
        ),
        "provider_calls_made_by_preflight": False,
        "epicure_calls_made_by_preflight": False,
        "official": False,
        "rank_eligible": False,
    }


def freeze(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    """Write all deterministic no-call v4 protocol artifacts."""

    history_path = _write_artifact(
        output_dir, "reasoning-effort-v4-history-audit", build_history_audit(repo_root)
    )
    history = _regular_json(history_path)
    baseline_path = _write_artifact(
        output_dir,
        "reasoning-effort-v4-low-baseline-audit",
        build_low_baseline_audit(repo_root),
    )
    baseline = _regular_json(baseline_path)
    route_payload = build_route_plan(
        repo_root,
        history_sha256=history["artifact_sha256"],
        baseline_sha256=baseline["artifact_sha256"],
    )
    route_path = _write_artifact(output_dir, "reasoning-effort-v4-route-gate-plan", route_payload)
    route = _regular_json(route_path)
    study_payload = build_study_plan(
        repo_root,
        history_sha256=history["artifact_sha256"],
        baseline=baseline,
        route_plan=route,
    )
    study_path = _write_artifact(output_dir, "reasoning-effort-v4-study-plan", study_payload)
    study = _regular_json(study_path)
    assets_payload = materialize_runner_assets(
        repo_root=repo_root, study=study, output_dir=output_dir / "runner"
    )
    assets_path = _write_artifact(output_dir, "reasoning-effort-v4-runner-assets", assets_payload)
    assets = _regular_json(assets_path)
    preflight_path = _write_artifact(
        output_dir,
        "reasoning-effort-v4-preflight",
        build_preflight(study=study, route_plan=route, runner_assets=assets),
    )
    return {
        "history": history_path,
        "baseline": baseline_path,
        "route_plan": route_path,
        "study_plan": study_path,
        "runner_assets": assets_path,
        "preflight": preflight_path,
    }


def verify_frozen(
    *,
    repo_root: Path,
    history_path: Path,
    baseline_path: Path,
    route_plan_path: Path,
    study_plan_path: Path,
    runner_assets_path: Path,
    preflight_path: Path,
) -> bool:
    """Re-open and rederive the full no-call protocol; hash-only summaries never pass."""

    try:
        history = _regular_json(history_path)
        baseline = _regular_json(baseline_path)
        route = _regular_json(route_plan_path)
        study = _regular_json(study_plan_path)
        assets = _regular_json(runner_assets_path)
        preflight = _regular_json(preflight_path)
        if not all(
            (
                _artifact_verifies(history, HISTORY_SCHEMA),
                _artifact_verifies(baseline, BASELINE_SCHEMA),
                _artifact_verifies(route, ROUTE_PLAN_SCHEMA),
                _artifact_verifies(study, STUDY_PLAN_SCHEMA),
                _artifact_verifies(assets, RUNNER_ASSETS_SCHEMA),
                _artifact_verifies(preflight, PREFLIGHT_SCHEMA),
            )
        ):
            return False
        expected_history = _payload_with_digest(build_history_audit(repo_root))
        expected_baseline = _payload_with_digest(build_low_baseline_audit(repo_root))
        if history != expected_history or baseline != expected_baseline:
            return False
        expected_route = _payload_with_digest(
            build_route_plan(
                repo_root,
                history_sha256=history["artifact_sha256"],
                baseline_sha256=baseline["artifact_sha256"],
            )
        )
        if route != expected_route:
            return False
        expected_study = _payload_with_digest(
            build_study_plan(
                repo_root,
                history_sha256=history["artifact_sha256"],
                baseline=baseline,
                route_plan=route,
            )
        )
        if study != expected_study:
            return False
        if assets.get("study_plan_sha256") != study["artifact_sha256"]:
            return False
        # Re-open every generated manifest/task file and enforce one-pair commands.
        for record in assets.get("variants") or []:
            if not isinstance(record, Mapping):
                return False
            manifest_path = _repo_path(repo_root, str(record.get("manifest") or ""))
            dossier_path = _repo_path(repo_root, str(record.get("task_dossier") or ""))
            manifest = _manifest(manifest_path)
            dossier = _regular_json(dossier_path)
            command = record.get("single_pair_dry_run_command")
            if (
                manifest["content_address"]["digest"] != record.get("manifest_sha256")
                or dossier.get("artifact_sha256") != record.get("task_dossier_sha256")
                or not _artifact_verifies(dossier, dossier.get("schema_version"))
                or not isinstance(command, list)
                or command.count("--max-new-pairs") != 1
                or command[command.index("--max-new-pairs") + 1] != "1"
            ):
                return False
        expected_preflight = _payload_with_digest(
            build_preflight(study=study, route_plan=route, runner_assets=assets)
        )
        return preflight == expected_preflight
    except (SensitivityV4Error, IntegrityError, KeyError, TypeError, ValueError):
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--repo-root", type=Path, required=True)
    for name in (
        "history",
        "baseline",
        "route-plan",
        "study-plan",
        "runner-assets",
        "preflight",
    ):
        verify_parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        records = freeze(args.repo_root.resolve(), args.output_dir.resolve())
        print(json.dumps({key: str(value) for key, value in records.items()}, indent=2))
        return
    passed = verify_frozen(
        repo_root=args.repo_root.resolve(),
        history_path=args.history,
        baseline_path=args.baseline,
        route_plan_path=args.route_plan,
        study_plan_path=args.study_plan,
        runner_assets_path=args.runner_assets,
        preflight_path=args.preflight,
    )
    print(json.dumps({"verified": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
