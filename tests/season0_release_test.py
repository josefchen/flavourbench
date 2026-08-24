from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_release import Season0ReleaseError, _member, build_release


def _artifact(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    return path


def _fixture(root: Path) -> dict[str, Path]:
    arm_payload = {"arm_id": "a"}
    arm_sha = sha256_json(arm_payload)
    target_event_payload = {"arm_id": "a", "event": "request_started"}
    correction_payload = {"correction_id": "a"}
    correction_sha = sha256_json(correction_payload)
    judgment_payload = {"judgment_id": "a"}
    judgment_sha = sha256_json(judgment_payload)
    judge_event_payload = {"judgment_id": "a", "event_type": "request_started"}
    task = _artifact(
        root / "inputs/task.json",
        {"synthetic_tasks": 0, "counts": {"total": 1, "synthetic": 0}},
    )
    task_sha = json.loads(task.read_bytes())["artifact_sha256"]
    epicure = _artifact(root / "inputs/epicure.json", {"release_id": "real-v1"})
    epicure_sha = json.loads(epicure.read_bytes())["artifact_sha256"]
    compatibility_payload = {"status": "real_contract_pass"}
    compatibility = _artifact(
        root
        / "compatibility"
        / f"compatibility-model-one-{sha256_json(compatibility_payload)}.json",
        compatibility_payload,
    )
    compatibility_sha = json.loads(compatibility.read_bytes())["artifact_sha256"]
    model = _artifact(
        root / "inputs/models.json",
        {
            "task_bank_artifact_sha256": task_sha,
            "epicure_intervention_artifact_sha256": epicure_sha,
            "counts": {"models": 1, "synthetic_models": 0, "placeholder_models": 0},
            "models": [
                {
                    "season_model_id": "model-1",
                    "compatibility_artifact_sha256": compatibility_sha,
                }
            ],
        },
    )
    model_sha = json.loads(model.read_bytes())["artifact_sha256"]
    comparison = _artifact(
        root / "inputs/comparisons.json",
        {
            "task_bank_artifact_sha256": task_sha,
            "model_manifest_artifact_sha256": model_sha,
            "synthetic_comparisons": 0,
        },
    )
    comparison_sha = json.loads(comparison.read_bytes())["artifact_sha256"]
    judge = _artifact(root / "inputs/judges.json", {"judges": [{"judge_id": "j1"}]})
    judge_sha = json.loads(judge.read_bytes())["artifact_sha256"]
    target_collection_summary = _artifact(
        root / "inputs/target-collection-summary.json",
        {
            "status": "collection_complete",
            "phase": "scored",
            "synthetic_arms": 0,
            "task_bank_artifact_sha256": task_sha,
            "model_manifest_artifact_sha256": model_sha,
            "epicure_intervention_artifact_sha256": epicure_sha,
            "counts": {"terminal_arms": 1},
            "arm_artifact_sha256s": [arm_sha],
        },
    )
    cost = _artifact(
        root / "inputs/cost.json",
        {
            "synthetic_arms": 0,
            "counts": {"arms": 1},
            "cost_correction_artifact_sha256s": [correction_sha],
        },
    )
    cost_sha = json.loads(cost.read_bytes())["artifact_sha256"]
    first_pass_summary = _artifact(
        root / "inputs/first-pass-summary.json",
        {
            "status": "collection_complete",
            "synthetic_judgments": 0,
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
            "judgment_artifact_sha256s": [judgment_sha],
        },
    )
    first_pass_sha = json.loads(first_pass_summary.read_bytes())["artifact_sha256"]
    recovery_plan = _artifact(
        root / "inputs/recovery-plan.json",
        {
            "synthetic_judgments": 0,
            "preference_outcomes_inspected": False,
            "original_collection_summary_artifact_sha256": first_pass_sha,
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
        },
    )
    recovery_plan_sha = json.loads(recovery_plan.read_bytes())["artifact_sha256"]
    summary = _artifact(
        root / "inputs/summary.json",
        {
            "status": "collection_complete",
            "synthetic_judgments": 0,
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
            "original_collection_summary_artifact_sha256": first_pass_sha,
            "recovery_plan_artifact_sha256": recovery_plan_sha,
            "counts": {
                "terminal_judgments": 1,
                "provider_attempt_records": 1,
            },
            "all_attempt_artifact_sha256s": [judgment_sha],
            "judgment_artifact_sha256s": [judgment_sha],
        },
    )
    analysis = _artifact(
        root / "analysis/analysis.json",
        {
            "synthetic_arms": 0,
            "synthetic_judgments": 0,
            "task_bank_artifact_sha256": task_sha,
            "model_manifest_artifact_sha256": model_sha,
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
            "target_cost_audit_artifact_sha256": cost_sha,
            "counts": {"scored_arms": 1, "judgment_records": 1},
            "implementation": {
                "source_sha256": {"ranking.py": "a" * 64},
            },
        },
    )
    analysis_document = json.loads(analysis.read_bytes())
    supersession_registry = root / "inputs/analysis-supersession.json"
    supersession_registry.write_text(
        json.dumps(
            {
                "schema_version": "flavourbench-analysis-supersession-v1",
                "disposition": (
                    "The active artifact is retained for audit. It is not an authorized "
                    "public benchmark release."
                ),
                "active_artifact": {
                    "path": "fixture/analysis.json",
                    "embedded_artifact_sha256": analysis_document["artifact_sha256"],
                    "file_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
                    "ranking_source_sha256": "a" * 64,
                },
                "superseded_artifacts": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    directories = {
        "arms_dir": root / "records/arms",
        "target_events_dir": root / "records/target-events",
        "target_cost_corrections_dir": root / "records/target-cost-corrections",
        "judgments_dir": root / "records/judgments",
        "judge_events_dir": root / "records/judge-events",
        "recovery_events_dir": root / "records/recovery-events",
        "source_dir": root / "source",
        "contracts_dir": root / "contracts",
    }
    for directory in directories.values():
        directory.mkdir(parents=True)
    _artifact(directories["arms_dir"] / "arm-a.json", arm_payload)
    _artifact(
        directories["target_events_dir"] / "event-a.json",
        target_event_payload,
    )
    _artifact(
        directories["target_cost_corrections_dir"] / "correction-a.json",
        correction_payload,
    )
    _artifact(
        directories["judgments_dir"] / "judgment-a.json",
        judgment_payload,
    )
    _artifact(
        directories["judge_events_dir"] / "event-j.json",
        judge_event_payload,
    )
    (directories["source_dir"] / "analysis.py").write_text("VALUE = 1\n")
    cache = directories["source_dir"] / "__pycache__"
    cache.mkdir()
    (cache / "analysis.pyc").write_bytes(b"nondeterministic cache")
    (directories["contracts_dir"] / "contract.json").write_text('{"version":1}\n')
    benchmark_card = root / "BENCHMARK_CARD.md"
    benchmark_card.write_text("# Benchmark card\n")
    pyproject = root / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fixture"\n')
    return {
        "task_bank_path": task,
        "model_manifest_path": model,
        "epicure_manifest_path": epicure,
        "comparison_manifest_path": comparison,
        "judge_manifest_path": judge,
        "target_collection_summary_path": target_collection_summary,
        "target_cost_audit_path": cost,
        "first_pass_judgment_summary_path": first_pass_summary,
        "recovery_plan_path": recovery_plan,
        "judgment_summary_path": summary,
        "analysis_path": analysis,
        "supersession_registry_path": supersession_registry,
        "analysis_dir": analysis.parent,
        "compatibility_root": compatibility.parent,
        "benchmark_card_path": benchmark_card,
        "pyproject_path": pyproject,
        **directories,
    }


def test_release_is_deterministic_bound_and_complete(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    first = build_release(**fixture, output_dir=tmp_path / "out-a")
    second = build_release(**fixture, output_dir=tmp_path / "out-b")
    first_archive = Path(first["archive_path"])
    second_archive = Path(second["archive_path"])
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["archive_sha256"] == hashlib.sha256(first_archive.read_bytes()).hexdigest()
    assert first["counts"]["target_arm_records"] == 1
    assert first["counts"]["judgment_provider_attempt_records"] == 1
    assert first["release_status"] == "internal_reproducibility_candidate_public_release_held"
    assert first["schema_version"] == "flavourbench-season0-research-release-v3"
    assert len(first["bindings"]["evidence_inventory"]) == 64
    assert len(first["bindings"]["supersession_registry_file"]) == 64
    with tarfile.open(first_archive, "r:gz") as archive:
        names = archive.getnames()
        assert names[0] == "release/MANIFEST.json"
        assert "records/target-arms/arm-a.json" in names
        assert "accounting/target-cost-corrections/correction-a.json" in names
        assert any(name.startswith("evidence/model-compatibility/") for name in names)
        assert "results/analysis.json" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_release_secret_scan_fails_closed(tmp_path: Path) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text('{"token":"AKIA1234567890ABCDEF"}\n')
    with pytest.raises(Season0ReleaseError, match="aws_access_key"):
        _member(secret, "unsafe/secret.json")


def test_release_secret_scan_distinguishes_a_detector_literal_from_a_key(
    tmp_path: Path,
) -> None:
    detector = tmp_path / "detector.py"
    detector.write_text('MARKER = "-----BEGIN PRIVATE KEY-----"\\n')
    _member(detector, "implementation/detector.py")
    key = tmp_path / "key.txt"
    key.write_text(
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
        "-----END PRIVATE KEY-----\n"
    )
    with pytest.raises(Season0ReleaseError, match="private_key"):
        _member(key, "unsafe/key.txt")


def test_release_rejects_tampered_content_address(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact.json", {"value": 1})
    document = json.loads(artifact.read_bytes())
    document["value"] = 2
    artifact.write_text(json.dumps(document))
    with pytest.raises(Season0ReleaseError, match="artifact hash mismatch"):
        _member(artifact, "records/artifact.json")


def test_release_rejects_analysis_listed_as_superseded(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    registry_path = fixture["supersession_registry_path"]
    registry = json.loads(registry_path.read_bytes())
    registry["superseded_artifacts"] = [
        {"embedded_artifact_sha256": registry["active_artifact"]["embedded_artifact_sha256"]}
    ]
    registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n")
    with pytest.raises(Season0ReleaseError, match="sole active artifact"):
        build_release(**fixture, output_dir=tmp_path / "out")


def test_release_rejects_same_count_arm_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    _artifact(fixture["arms_dir"] / "arm-a.json", {"arm_id": "substituted"})
    with pytest.raises(Season0ReleaseError, match="exact hash registry"):
        build_release(**fixture, output_dir=tmp_path / "out")


def test_release_rejects_same_count_judgment_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    _artifact(
        fixture["judgments_dir"] / "judgment-a.json",
        {"judgment_id": "substituted"},
    )
    with pytest.raises(Season0ReleaseError, match="exact hash registry"):
        build_release(**fixture, output_dir=tmp_path / "out")


def test_release_rejects_cost_correction_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    _artifact(
        fixture["target_cost_corrections_dir"] / "correction-a.json",
        {"correction_id": "substituted"},
    )
    with pytest.raises(Season0ReleaseError, match="exact hash registry"):
        build_release(**fixture, output_dir=tmp_path / "out")


def test_release_rejects_request_event_identity_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    _artifact(
        fixture["target_events_dir"] / "event-a.json",
        {"arm_id": "substituted", "event": "request_started"},
    )
    with pytest.raises(Season0ReleaseError, match="request-event identities"):
        build_release(**fixture, output_dir=tmp_path / "out")


def test_release_rejects_unexpected_analysis_json(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    _artifact(fixture["analysis_dir"] / "other-analysis.json", {"value": 1})
    with pytest.raises(Season0ReleaseError, match="unexpected JSON artifact"):
        build_release(**fixture, output_dir=tmp_path / "out")
