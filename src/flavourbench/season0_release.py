"""Build a deterministic, secret-scanned FlavourBench Season 0 research release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .real_task_bank import sha256_json
from .season0_arm_corrections import validate_arm_interpretation_correction

SCHEMA_VERSION = "flavourbench-season0-research-release-v3"
SOURCE_DATE_EPOCH = 1_784_160_000
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".csv"}
SECRET_PATTERNS = {
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "openrouter_api_key": re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----[ \t]*\r?\n"),
    "bearer_token": re.compile(
        rb"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/-]{20,}"
    ),
    "secret_field": re.compile(
        rb"(?i)[\"']?(?:aws_secret_access_key|openrouter_api_key|secret_access_key)"
        rb"[\"']?\s*[:=]\s*[\"'][^\"']{12,}[\"']"
    ),
}


class Season0ReleaseError(RuntimeError):
    """The requested release is incomplete, unsafe, or not the frozen Season 0 run."""


@dataclass(frozen=True)
class ReleaseMember:
    logical_path: str
    source_path: Path
    sha256: str
    size_bytes: int


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Season0ReleaseError(f"{path} is not a JSON object")
    return value


def _verify(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise Season0ReleaseError(f"{label} artifact hash mismatch")
    return actual


def _safe_logical_path(value: str) -> str:
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise Season0ReleaseError(f"unsafe release path: {value}")
    return logical.as_posix()


def _scan_secrets(path: Path, data: bytes) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(data):
            raise Season0ReleaseError(f"secret scan matched {label} in {path}")


def _member(source: Path, logical_path: str) -> ReleaseMember:
    if source.is_symlink() or not source.is_file():
        raise Season0ReleaseError(f"release source is not a regular file: {source}")
    data = source.read_bytes()
    _scan_secrets(source, data)
    if source.suffix.lower() == ".json":
        try:
            document = json.loads(data)
        except json.JSONDecodeError as error:
            raise Season0ReleaseError(f"invalid JSON release member: {source}") from error
        if isinstance(document, Mapping) and "artifact_sha256" in document:
            _verify(document, str(source))
    return ReleaseMember(
        logical_path=_safe_logical_path(logical_path),
        source_path=source,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _tree_members(source: Path, logical_root: str) -> list[ReleaseMember]:
    if not source.is_dir():
        raise Season0ReleaseError(f"release tree is missing: {source}")
    output = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (
            any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
            or path.suffix.lower() in {".pyc", ".pyo"}
            or not (path.is_file() or path.is_symlink())
        ):
            continue
        output.append(_member(path, f"{logical_root}/{relative.as_posix()}"))
    return output


def _required_hashes(document: Mapping[str, Any], key: str, label: str) -> list[str]:
    values = document.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in values
        )
        or len(values) != len(set(values))
    ):
        raise Season0ReleaseError(f"{label} has an invalid {key} registry")
    return values


def _artifact_documents(members: Sequence[ReleaseMember], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for member in members:
        document = _load(member.source_path)
        digest = document.get("artifact_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise Season0ReleaseError(f"{label} record has no artifact hash")
        if digest in output:
            raise Season0ReleaseError(f"{label} contains a duplicate artifact hash")
        output[digest] = document
    return output


def _assert_exact_registry(
    *, members: Sequence[ReleaseMember], expected: Sequence[str], label: str
) -> dict[str, dict[str, Any]]:
    documents = _artifact_documents(members, label)
    if set(documents) != set(expected) or len(documents) != len(expected):
        missing = len(set(expected) - set(documents))
        extra = len(set(documents) - set(expected))
        raise Season0ReleaseError(
            f"{label} differs from its exact hash registry (missing={missing}, extra={extra})"
        )
    return documents


def _identity_map(
    documents: Mapping[str, Mapping[str, Any]], field: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for document in documents.values():
        identity = document.get(field)
        if not isinstance(identity, str) or not identity or identity in output:
            raise Season0ReleaseError(f"{label} has a missing or duplicate {field}")
        output[identity] = document
    return output


def _validate_supersession_registry(
    *,
    registry: Mapping[str, Any],
    registry_path: Path,
    analysis: Mapping[str, Any],
    analysis_path: Path,
    arm_interpretation_correction: Mapping[str, Any] | None = None,
    arm_interpretation_correction_path: Path | None = None,
) -> str:
    schema_version = registry.get("schema_version")
    if schema_version not in {
        "flavourbench-analysis-supersession-v1",
        "flavourbench-analysis-supersession-v2",
    }:
        raise Season0ReleaseError("unsupported analysis supersession registry")
    active = registry.get("active_artifact")
    if schema_version == "flavourbench-analysis-supersession-v1":
        superseded = registry.get("superseded_artifacts")
    else:
        directly_superseded = registry.get("directly_superseded_artifact")
        superseded = [directly_superseded] if isinstance(directly_superseded, Mapping) else None
    if not isinstance(active, Mapping) or not isinstance(superseded, list):
        raise Season0ReleaseError("analysis supersession registry is malformed")
    analysis_sha = str(analysis.get("artifact_sha256") or "")
    active_sha = str(active.get("embedded_artifact_sha256") or "")
    superseded_hashes = {
        str(row.get("embedded_artifact_sha256") or "")
        for row in superseded
        if isinstance(row, Mapping)
    }
    if (
        not re.fullmatch(r"[0-9a-f]{64}", active_sha)
        or analysis_sha != active_sha
        or analysis_sha in superseded_hashes
    ):
        raise Season0ReleaseError(
            "analysis is not the supersession registry's sole active artifact"
        )
    file_sha = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    if active.get("file_sha256") != file_sha:
        raise Season0ReleaseError("active analysis byte hash differs from supersession registry")
    registered_path = active.get("path")
    if not isinstance(registered_path, str) or Path(registered_path).name != analysis_path.name:
        raise Season0ReleaseError("active analysis path differs from supersession registry")
    implementation = analysis.get("implementation")
    source_hashes = (
        implementation.get("source_sha256") if isinstance(implementation, Mapping) else None
    )
    if not isinstance(source_hashes, Mapping) or source_hashes.get("ranking.py") != active.get(
        "ranking_source_sha256"
    ):
        raise Season0ReleaseError(
            "active ranking implementation differs from supersession registry"
        )
    disposition = registry.get("disposition")
    if (
        not isinstance(disposition, str)
        or "not an authorized public benchmark release" not in disposition
    ):
        raise Season0ReleaseError("supersession registry lacks the public-release hold")
    if schema_version == "flavourbench-analysis-supersession-v2":
        binding = registry.get("correction_binding")
        if (
            not isinstance(binding, Mapping)
            or arm_interpretation_correction is None
            or arm_interpretation_correction_path is None
            or binding.get("embedded_artifact_sha256")
            != arm_interpretation_correction.get("artifact_sha256")
            or binding.get("file_sha256")
            != hashlib.sha256(arm_interpretation_correction_path.read_bytes()).hexdigest()
            or binding.get("source_arm_set_sha256")
            != arm_interpretation_correction.get("source_arm_set_sha256")
            or binding.get("target_cost_audit_artifact_sha256")
            != analysis.get("target_cost_audit_artifact_sha256")
        ):
            raise Season0ReleaseError(
                "analysis supersession registry has an invalid correction binding"
            )
    return hashlib.sha256(registry_path.read_bytes()).hexdigest()


def _compatibility_members(source: Path, model_manifest: Mapping[str, Any]) -> list[ReleaseMember]:
    if not source.is_dir():
        raise Season0ReleaseError(f"compatibility artifact root is missing: {source}")
    output = []
    seen: set[str] = set()
    for model in model_manifest.get("models", []):
        if not isinstance(model, Mapping):
            raise Season0ReleaseError("model manifest contains a malformed system")
        digest = str(model.get("compatibility_artifact_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in seen:
            raise Season0ReleaseError("model compatibility hashes are missing or duplicated")
        seen.add(digest)
        matches = sorted(source.rglob(f"*{digest}*.json"))
        if len(matches) != 1:
            raise Season0ReleaseError(
                f"expected one compatibility artifact for {digest}; found {len(matches)}"
            )
        member = _member(matches[0], f"evidence/model-compatibility/{matches[0].name}")
        if json.loads(matches[0].read_bytes()).get("artifact_sha256") != digest:
            raise Season0ReleaseError("compatibility artifact hash differs from manifest")
        output.append(member)
    if not output:
        raise Season0ReleaseError("model manifest has no compatibility evidence")
    return output


def _validate_raw_evidence(
    *,
    arm_members: Sequence[ReleaseMember],
    target_event_members: Sequence[ReleaseMember],
    cost_correction_members: Sequence[ReleaseMember],
    judgment_members: Sequence[ReleaseMember],
    judge_event_members: Sequence[ReleaseMember],
    recovery_event_members: Sequence[ReleaseMember],
    target_collection_summary: Mapping[str, Any],
    target_cost_audit: Mapping[str, Any],
    first_pass_judgment_summary: Mapping[str, Any],
    recovery_plan: Mapping[str, Any],
    judgment_summary: Mapping[str, Any],
) -> str:
    arm_hashes = _required_hashes(
        target_collection_summary,
        "arm_artifact_sha256s",
        "target collection summary",
    )
    first_pass_hashes = _required_hashes(
        first_pass_judgment_summary,
        "judgment_artifact_sha256s",
        "first-pass judgment summary",
    )
    all_attempt_hashes = _required_hashes(
        judgment_summary,
        "all_attempt_artifact_sha256s",
        "final judgment summary",
    )
    terminal_hashes = _required_hashes(
        judgment_summary,
        "judgment_artifact_sha256s",
        "final judgment summary",
    )
    correction_hashes = _required_hashes(
        target_cost_audit,
        "cost_correction_artifact_sha256s",
        "target cost audit",
    )

    arms = _assert_exact_registry(members=arm_members, expected=arm_hashes, label="target arms")
    corrections = _assert_exact_registry(
        members=cost_correction_members,
        expected=correction_hashes,
        label="target cost corrections",
    )
    judgments = _assert_exact_registry(
        members=judgment_members,
        expected=all_attempt_hashes,
        label="judgment attempts",
    )
    if not set(first_pass_hashes) <= set(all_attempt_hashes):
        raise Season0ReleaseError("first-pass judgment registry is not an attempt subset")
    if not set(terminal_hashes) <= set(all_attempt_hashes):
        raise Season0ReleaseError("terminal judgment registry is not an attempt subset")

    arm_by_id = _identity_map(arms, "arm_id", "target arms")
    target_events = _artifact_documents(target_event_members, "target request events")
    target_event_by_id = _identity_map(target_events, "arm_id", "target request events")
    if set(target_event_by_id) != set(arm_by_id) or any(
        event.get("event") != "request_started" for event in target_events.values()
    ):
        raise Season0ReleaseError("target request-event identities do not match target arms")

    first_pass = {digest: judgments[digest] for digest in first_pass_hashes}
    first_pass_by_id = _identity_map(first_pass, "judgment_id", "first-pass judgments")
    judge_events = _artifact_documents(judge_event_members, "judge request events")
    judge_event_by_id = _identity_map(judge_events, "judgment_id", "judge request events")
    if set(judge_event_by_id) != set(first_pass_by_id) or any(
        event.get("event_type") != "request_started" for event in judge_events.values()
    ):
        raise Season0ReleaseError(
            "judge request-event identities do not match first-pass judgments"
        )

    recovery_hashes = set(all_attempt_hashes) - set(first_pass_hashes)
    recovery = {digest: judgments[digest] for digest in recovery_hashes}
    recovery_by_id = _identity_map(recovery, "judgment_id", "recovery judgments")
    recovery_events = _artifact_documents(recovery_event_members, "judge recovery request events")
    recovery_event_by_id = _identity_map(
        recovery_events, "judgment_id", "judge recovery request events"
    )
    if set(recovery_event_by_id) != set(recovery_by_id):
        raise Season0ReleaseError(
            "recovery request-event identities do not match recovery attempts"
        )
    plan_sha = recovery_plan.get("artifact_sha256")
    for judgment_id, document in recovery_by_id.items():
        original = first_pass_by_id.get(judgment_id)
        event = recovery_event_by_id[judgment_id]
        original_sha = original.get("artifact_sha256") if isinstance(original, Mapping) else None
        if (
            not isinstance(original_sha, str)
            or document.get("supersedes_artifact_sha256") != original_sha
            or event.get("supersedes_artifact_sha256") != original_sha
            or event.get("plan_artifact_sha256") != plan_sha
            or event.get("event_type") != "recovery_request_started"
        ):
            raise Season0ReleaseError("recovery supersession chain is invalid")

    terminal = {digest: judgments[digest] for digest in terminal_hashes}
    terminal_by_id = _identity_map(terminal, "judgment_id", "terminal judgments")
    if set(terminal_by_id) != set(first_pass_by_id):
        raise Season0ReleaseError("terminal judgment identities are incomplete")

    inventory = {
        "target_arms": sorted(
            (identity, str(document["artifact_sha256"])) for identity, document in arm_by_id.items()
        ),
        "target_events": sorted(
            (identity, str(document["artifact_sha256"]))
            for identity, document in target_event_by_id.items()
        ),
        "target_cost_corrections": sorted(corrections),
        "judgment_attempts": sorted(
            (str(document["judgment_id"]), digest) for digest, document in judgments.items()
        ),
        "terminal_judgments": sorted(
            (identity, str(document["artifact_sha256"]))
            for identity, document in terminal_by_id.items()
        ),
        "judge_events": sorted(
            (identity, str(document["artifact_sha256"]))
            for identity, document in judge_event_by_id.items()
        ),
        "recovery_events": sorted(
            (identity, str(document["artifact_sha256"]))
            for identity, document in recovery_event_by_id.items()
        ),
    }
    return sha256_json(inventory)


def _validate_bindings(
    *,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    epicure_manifest: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    target_collection_summary: Mapping[str, Any],
    target_cost_audit: Mapping[str, Any],
    first_pass_judgment_summary: Mapping[str, Any],
    recovery_plan: Mapping[str, Any],
    judgment_summary: Mapping[str, Any],
    analysis: Mapping[str, Any],
    arm_interpretation_correction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    hashes = {
        "task_bank": _verify(task_bank, "task bank"),
        "model_manifest": _verify(model_manifest, "model manifest"),
        "epicure_manifest": _verify(epicure_manifest, "Epicure intervention"),
        "comparison_manifest": _verify(comparison_manifest, "comparison manifest"),
        "judge_manifest": _verify(judge_manifest, "judge manifest"),
        "target_collection_summary": _verify(
            target_collection_summary, "target collection summary"
        ),
        "target_cost_audit": _verify(target_cost_audit, "target cost audit"),
        "first_pass_judgment_summary": _verify(
            first_pass_judgment_summary, "first-pass judgment summary"
        ),
        "recovery_plan": _verify(recovery_plan, "judge recovery plan"),
        "judgment_summary": _verify(judgment_summary, "judgment summary"),
        "analysis": _verify(analysis, "analysis"),
    }
    if arm_interpretation_correction is not None:
        hashes["arm_interpretation_correction"] = _verify(
            arm_interpretation_correction,
            "arm interpretation correction",
        )
    expected_analysis = {
        "task_bank_artifact_sha256": hashes["task_bank"],
        "model_manifest_artifact_sha256": hashes["model_manifest"],
        "comparison_manifest_artifact_sha256": hashes["comparison_manifest"],
        "judge_manifest_artifact_sha256": hashes["judge_manifest"],
        "target_cost_audit_artifact_sha256": hashes["target_cost_audit"],
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise Season0ReleaseError("analysis is bound to another frozen experiment")
    correction_sha = hashes.get("arm_interpretation_correction")
    if (
        analysis.get("arm_interpretation_correction_artifact_sha256") != correction_sha
        or target_cost_audit.get("arm_interpretation_correction_artifact_sha256") != correction_sha
    ):
        raise Season0ReleaseError("analysis or cost audit lacks the active arm correction")
    if model_manifest.get("epicure_intervention_artifact_sha256") != hashes["epicure_manifest"]:
        raise Season0ReleaseError("model manifest is bound to another Epicure intervention")
    if (
        target_collection_summary.get("status") != "collection_complete"
        or target_collection_summary.get("phase") != "scored"
        or target_collection_summary.get("synthetic_arms") != 0
        or target_collection_summary.get("task_bank_artifact_sha256") != hashes["task_bank"]
        or target_collection_summary.get("model_manifest_artifact_sha256")
        != hashes["model_manifest"]
        or target_collection_summary.get("epicure_intervention_artifact_sha256")
        != hashes["epicure_manifest"]
    ):
        raise Season0ReleaseError("target collection summary is incomplete or misbound")
    if (
        judgment_summary.get("status") != "collection_complete"
        or first_pass_judgment_summary.get("status") != "collection_complete"
        or first_pass_judgment_summary.get("synthetic_judgments") != 0
        or recovery_plan.get("synthetic_judgments") != 0
        or recovery_plan.get("preference_outcomes_inspected") is not False
        or judgment_summary.get("comparison_manifest_artifact_sha256")
        != hashes["comparison_manifest"]
        or judgment_summary.get("judge_manifest_artifact_sha256") != hashes["judge_manifest"]
        or judgment_summary.get("original_collection_summary_artifact_sha256")
        != hashes["first_pass_judgment_summary"]
        or judgment_summary.get("recovery_plan_artifact_sha256") != hashes["recovery_plan"]
        or recovery_plan.get("original_collection_summary_artifact_sha256")
        != hashes["first_pass_judgment_summary"]
        or recovery_plan.get("comparison_manifest_artifact_sha256") != hashes["comparison_manifest"]
        or recovery_plan.get("judge_manifest_artifact_sha256") != hashes["judge_manifest"]
    ):
        raise Season0ReleaseError("judgment summary is incomplete or misbound")
    summary_counts = judgment_summary.get("counts")
    analysis_counts = analysis.get("counts")
    if not isinstance(summary_counts, Mapping) or not isinstance(analysis_counts, Mapping):
        raise Season0ReleaseError("release counts are missing")
    if int(summary_counts.get("terminal_judgments") or 0) != int(
        analysis_counts.get("judgment_records") or 0
    ):
        raise Season0ReleaseError("judgment identity counts do not reconcile")
    target_counts = target_collection_summary.get("counts")
    if (
        not isinstance(target_counts, Mapping)
        or int(target_counts.get("terminal_arms") or 0)
        != int(analysis_counts.get("scored_arms") or 0)
        or int(target_cost_audit.get("counts", {}).get("arms") or 0)
        != int(analysis_counts.get("scored_arms") or 0)
    ):
        raise Season0ReleaseError("target collection counts do not reconcile")
    if (
        task_bank.get("synthetic_tasks") != 0
        or task_bank.get("counts", {}).get("synthetic") != 0
        or model_manifest.get("counts", {}).get("synthetic_models") != 0
        or model_manifest.get("counts", {}).get("placeholder_models") != 0
        or comparison_manifest.get("synthetic_comparisons") != 0
        or target_cost_audit.get("synthetic_arms") != 0
        or judgment_summary.get("synthetic_judgments") != 0
        or analysis.get("synthetic_arms") != 0
        or analysis.get("synthetic_judgments") != 0
    ):
        raise Season0ReleaseError("official release refuses synthetic or placeholder evidence")
    return hashes


def _manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = SOURCE_DATE_EPOCH
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(*, members: Sequence[ReleaseMember], manifest: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                fileobj=raw, mode="wb", filename="", mtime=SOURCE_DATE_EPOCH
            ) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    archive.addfile(
                        _tar_info("release/MANIFEST.json", len(manifest)),
                        io.BytesIO(manifest),
                    )
                    for member in sorted(members, key=lambda item: item.logical_path):
                        data = member.source_path.read_bytes()
                        if hashlib.sha256(data).hexdigest() != member.sha256:
                            raise Season0ReleaseError(
                                f"release input changed during packaging: {member.source_path}"
                            )
                        archive.addfile(_tar_info(member.logical_path, len(data)), io.BytesIO(data))
        os.chmod(temporary, 0o644)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_release(
    *,
    task_bank_path: Path,
    model_manifest_path: Path,
    epicure_manifest_path: Path,
    comparison_manifest_path: Path,
    judge_manifest_path: Path,
    target_collection_summary_path: Path,
    target_cost_audit_path: Path,
    first_pass_judgment_summary_path: Path,
    recovery_plan_path: Path,
    judgment_summary_path: Path,
    analysis_path: Path,
    supersession_registry_path: Path,
    arms_dir: Path,
    target_events_dir: Path,
    target_cost_corrections_dir: Path,
    judgments_dir: Path,
    judge_events_dir: Path,
    recovery_events_dir: Path,
    analysis_dir: Path,
    compatibility_root: Path,
    source_dir: Path,
    contracts_dir: Path,
    benchmark_card_path: Path,
    pyproject_path: Path,
    output_dir: Path,
    arm_interpretation_correction_path: Path | None = None,
) -> dict[str, Any]:
    documents = {
        "task_bank": _load(task_bank_path),
        "model_manifest": _load(model_manifest_path),
        "epicure_manifest": _load(epicure_manifest_path),
        "comparison_manifest": _load(comparison_manifest_path),
        "judge_manifest": _load(judge_manifest_path),
        "target_collection_summary": _load(target_collection_summary_path),
        "target_cost_audit": _load(target_cost_audit_path),
        "first_pass_judgment_summary": _load(first_pass_judgment_summary_path),
        "recovery_plan": _load(recovery_plan_path),
        "judgment_summary": _load(judgment_summary_path),
        "analysis": _load(analysis_path),
    }
    if arm_interpretation_correction_path is not None:
        documents["arm_interpretation_correction"] = _load(arm_interpretation_correction_path)
    hashes = _validate_bindings(**documents)
    validated_correction = validate_arm_interpretation_correction(
        correction=documents.get("arm_interpretation_correction"),
        arms_dir=arms_dir,
    )
    if validated_correction is not None:
        hashes["arm_interpretation_correction_source_arm_set"] = (
            validated_correction.source_arm_set_sha256
        )
    hashes["supersession_registry_file"] = _validate_supersession_registry(
        registry=_load(supersession_registry_path),
        registry_path=supersession_registry_path,
        analysis=documents["analysis"],
        analysis_path=analysis_path,
        arm_interpretation_correction=documents.get("arm_interpretation_correction"),
        arm_interpretation_correction_path=arm_interpretation_correction_path,
    )
    if analysis_path.parent.resolve() != analysis_dir.resolve():
        raise Season0ReleaseError("analysis path is outside the declared analysis directory")
    other_analysis_files = [
        path for path in analysis_dir.glob("*.json") if path.resolve() != analysis_path.resolve()
    ]
    if other_analysis_files:
        raise Season0ReleaseError("analysis directory contains an unexpected JSON artifact")
    fixed = [
        _member(task_bank_path, "data/task-bank.json"),
        _member(model_manifest_path, "manifests/model-manifest.json"),
        _member(epicure_manifest_path, "manifests/epicure-intervention.json"),
        _member(comparison_manifest_path, "manifests/comparisons.json"),
        _member(judge_manifest_path, "manifests/judges.json"),
        _member(
            target_collection_summary_path,
            "accounting/target-collection-summary.json",
        ),
        _member(target_cost_audit_path, "accounting/target-cost-audit.json"),
        _member(
            first_pass_judgment_summary_path,
            "accounting/judgment-first-pass-summary.json",
        ),
        _member(recovery_plan_path, "governance/judge-throttle-recovery-plan.json"),
        _member(judgment_summary_path, "accounting/judgment-final-summary.json"),
        _member(analysis_path, "results/analysis.json"),
        _member(
            supersession_registry_path,
            "governance/analysis-supersession-registry.json",
        ),
        _member(benchmark_card_path, "documentation/BENCHMARK_CARD.md"),
        _member(pyproject_path, "implementation/pyproject.toml"),
    ]
    if arm_interpretation_correction_path is not None:
        fixed.append(
            _member(
                arm_interpretation_correction_path,
                "governance/arm-interpretation-correction.json",
            )
        )
    arm_members = _tree_members(arms_dir, "records/target-arms")
    target_event_members = _tree_members(target_events_dir, "records/target-events")
    cost_correction_members = _tree_members(
        target_cost_corrections_dir, "accounting/target-cost-corrections"
    )
    judgment_members = _tree_members(judgments_dir, "records/judgments")
    judge_event_members = _tree_members(judge_events_dir, "records/judge-events")
    recovery_event_members = _tree_members(recovery_events_dir, "records/recovery-events")
    evidence_inventory_sha = _validate_raw_evidence(
        arm_members=arm_members,
        target_event_members=target_event_members,
        cost_correction_members=cost_correction_members,
        judgment_members=judgment_members,
        judge_event_members=judge_event_members,
        recovery_event_members=recovery_event_members,
        target_collection_summary=documents["target_collection_summary"],
        target_cost_audit=documents["target_cost_audit"],
        first_pass_judgment_summary=documents["first_pass_judgment_summary"],
        recovery_plan=documents["recovery_plan"],
        judgment_summary=documents["judgment_summary"],
    )
    hashes["evidence_inventory"] = evidence_inventory_sha
    trees = [
        *arm_members,
        *target_event_members,
        *cost_correction_members,
        *judgment_members,
        *judge_event_members,
        *recovery_event_members,
        *_compatibility_members(compatibility_root, documents["model_manifest"]),
        *_tree_members(source_dir, "implementation/src/flavourbench"),
        *_tree_members(contracts_dir, "implementation/contracts"),
    ]
    members = fixed + trees
    logical_paths = [member.logical_path for member in members]
    if len(logical_paths) != len(set(logical_paths)):
        raise Season0ReleaseError("release contains duplicate logical paths")
    arm_files = len(arm_members)
    judgment_files = len(judgment_members)
    summary_counts = documents["judgment_summary"]["counts"]
    analysis_counts = documents["analysis"]["counts"]
    if arm_files != int(analysis_counts.get("scored_arms") or 0):
        raise Season0ReleaseError("raw target-arm file count does not match analysis")
    if judgment_files != int(summary_counts.get("provider_attempt_records") or 0):
        raise Season0ReleaseError("raw judgment attempt count does not match final summary")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "release_status": "internal_reproducibility_candidate_public_release_held",
        "synthetic_tasks": 0,
        "synthetic_arms": 0,
        "synthetic_judgments": 0,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "bindings": hashes,
        "counts": {
            "tasks": int(documents["task_bank"]["counts"]["total"]),
            "models": int(documents["model_manifest"]["counts"]["models"]),
            "target_arm_records": arm_files,
            "judgment_identities": int(summary_counts["terminal_judgments"]),
            "judgment_provider_attempt_records": judgment_files,
            "release_members_excluding_manifest": len(members),
        },
        "privacy": {
            "participant_prompts": False,
            "raw_ip_addresses": False,
            "provider_credentials": False,
            "secret_scan": "passed",
            "task_source": "licensed public Seasoned Advice posts with retained attribution",
        },
        "members": [
            {
                "path": member.logical_path,
                "sha256": member.sha256,
                "size_bytes": member.size_bytes,
            }
            for member in sorted(members, key=lambda item: item.logical_path)
        ],
    }
    manifest = _manifest_bytes(payload)
    release_sha = sha256_json(payload)
    archive = output_dir / f"flavourbench-season0-research-release-{release_sha}.tar.gz"
    _write_archive(members=members, manifest=manifest, destination=archive)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_path = output_dir / "SHA256SUMS"
    checksum_data = f"{archive_sha}  {archive.name}\n".encode()
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(checksum_data)
        handle.flush()
    os.chmod(temporary, 0o644)
    temporary.replace(checksum_path)
    return {
        **payload,
        "artifact_sha256": release_sha,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "checksum_path": str(checksum_path),
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "task-bank",
        "model-manifest",
        "epicure-manifest",
        "comparison-manifest",
        "judge-manifest",
        "target-collection-summary",
        "target-cost-audit",
        "first-pass-judgment-summary",
        "recovery-plan",
        "judgment-summary",
        "analysis",
        "supersession-registry",
        "arm-interpretation-correction",
        "arms-dir",
        "target-events-dir",
        "target-cost-corrections-dir",
        "judgments-dir",
        "judge-events-dir",
        "recovery-events-dir",
        "analysis-dir",
        "compatibility-root",
        "source-dir",
        "contracts-dir",
        "benchmark-card",
        "pyproject",
        "output-dir",
    ):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_release(
        task_bank_path=args.task_bank,
        model_manifest_path=args.model_manifest,
        epicure_manifest_path=args.epicure_manifest,
        comparison_manifest_path=args.comparison_manifest,
        judge_manifest_path=args.judge_manifest,
        target_collection_summary_path=args.target_collection_summary,
        target_cost_audit_path=args.target_cost_audit,
        first_pass_judgment_summary_path=args.first_pass_judgment_summary,
        recovery_plan_path=args.recovery_plan,
        judgment_summary_path=args.judgment_summary,
        analysis_path=args.analysis,
        supersession_registry_path=args.supersession_registry,
        arm_interpretation_correction_path=args.arm_interpretation_correction,
        arms_dir=args.arms_dir,
        target_events_dir=args.target_events_dir,
        target_cost_corrections_dir=args.target_cost_corrections_dir,
        judgments_dir=args.judgments_dir,
        judge_events_dir=args.judge_events_dir,
        recovery_events_dir=args.recovery_events_dir,
        analysis_dir=args.analysis_dir,
        compatibility_root=args.compatibility_root,
        source_dir=args.source_dir,
        contracts_dir=args.contracts_dir,
        benchmark_card_path=args.benchmark_card,
        pyproject_path=args.pyproject,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"members"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
