"""Bind the exact v1 sampling frame to corrected human-study documentation.

V2 is an append-only, offline, blocked successor.  It inherits every comparison,
abstract rater slot, and concealed repeat from the immutable v1 artifact while
binding the corrected 3,200-primary/400-repeat documentation and checksum package.
It performs no network, provider, model, reviewer, or human-contact operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import task_campaign_human_sampling_successor as v1
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season1-human-judgment-sampling-v2-candidate"
STATUS = "blocked_documentation_bound_sampling_successor_not_authorized"

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAPER_ROOT = FLAVOURBENCH_ROOT.parent

V1_SEMANTIC_SHA256 = "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8"
V1_PHYSICAL_SHA256 = "6c7371daa9506cdf5dcee38c19ee48dd2938f36d259c1ee04b7c849c49977039"
V1_RECIPE_SHA256 = "b63c7955b74d20175602145db402cc1e1391c1ac3a179b0e443314ba0aae6600"

HUMAN_REVIEW_PHYSICAL_SHA256 = (
    "a326b47a752163cb27df8bcfab7baaa8b1e2d308af7a03d43254e33359cf468b"
)
STUDY_YAML_PHYSICAL_SHA256 = (
    "162e2b68300705c8721835b24df431c95676171e363f0a9c3949f28b8a0c63ef"
)
READINESS_V1_PHYSICAL_SHA256 = (
    "33c5fd525dc5f071fef7c829dc81ae9d9063ed6b8ccc16b0fdf9f05cc198b8ee"
)
READINESS_V2_PHYSICAL_SHA256 = (
    "8dddb1def07eac8ac7d23cc146903af865852cccd8442538c44499e357ea215f"
)
SUPERSESSION_PHYSICAL_SHA256 = (
    "7e7fe118ab50011e1b55833c733ab68ffeb9077b48bc1fb584a6b912606593ed"
)
GO_PACKAGE_V1_PHYSICAL_SHA256 = (
    "2d383474732dd4d4e4ad9a67e71292b2a0aaf6ae3f20dc5213b44dd9301841ad"
)
GO_PACKAGE_V2_PHYSICAL_SHA256 = (
    "49d8ccc88cf1d20d12d250b189534ba85aa7bb14410edb92fb3c902e06607195"
)

V1_REFERENCE = (
    "flavourbench/artifacts/season1/human-judgment-sampling-v1-candidate/"
    f"human-judgment-sampling-v1-candidate-{V1_SEMANTIC_SHA256}.json"
)
HUMAN_REVIEW_REFERENCE = "protocol/FLAVOURBENCH-HUMAN-REVIEW.md"
STUDY_YAML_REFERENCE = "protocol/study.yaml"
READINESS_V1_REFERENCE = (
    "governance/reviews/FLAVOURBENCH-HUMAN-STUDY-GO-READINESS-20260809.md"
)
READINESS_V2_REFERENCE = (
    "governance/reviews/FLAVOURBENCH-HUMAN-STUDY-GO-READINESS-v2-20260809.md"
)
SUPERSESSION_REFERENCE = (
    "governance/decisions/FLAVOURBENCH-HUMAN-WORKLOAD-SUPERSESSION-20260809.md"
)
GO_PACKAGE_V1_REFERENCE = "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v1.sha256"
GO_PACKAGE_V2_REFERENCE = "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v2.sha256"

DEFAULT_V1_ARTIFACT = EVALUATION_PAPER_ROOT / V1_REFERENCE
DEFAULT_HUMAN_REVIEW = EVALUATION_PAPER_ROOT / HUMAN_REVIEW_REFERENCE
DEFAULT_STUDY_YAML = EVALUATION_PAPER_ROOT / STUDY_YAML_REFERENCE
DEFAULT_READINESS_V1 = EVALUATION_PAPER_ROOT / READINESS_V1_REFERENCE
DEFAULT_READINESS_V2 = EVALUATION_PAPER_ROOT / READINESS_V2_REFERENCE
DEFAULT_SUPERSESSION = EVALUATION_PAPER_ROOT / SUPERSESSION_REFERENCE
DEFAULT_GO_PACKAGE_V1 = EVALUATION_PAPER_ROOT / GO_PACKAGE_V1_REFERENCE
DEFAULT_GO_PACKAGE_V2 = EVALUATION_PAPER_ROOT / GO_PACKAGE_V2_REFERENCE
DEFAULT_OUTPUT_DIR = FLAVOURBENCH_ROOT / (
    "artifacts/season1/human-judgment-sampling-v2-candidate"
)


class HumanSamplingV2Error(RuntimeError):
    """A v1 byte, corrected source, checksum, or no-replace write failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanSamplingV2Error(message)


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HumanSamplingV2Error(f"cannot open regular source: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"source is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_bound_bytes(path: Path, expected_sha256: str) -> bytes:
    data = _read_regular_bytes(path)
    _require(
        hashlib.sha256(data).hexdigest() == expected_sha256,
        f"physical digest mismatch: {path}",
    )
    return data


def _decode_bound_text(path: Path, expected_sha256: str) -> str:
    try:
        return _read_bound_bytes(path, expected_sha256).decode("utf-8")
    except UnicodeDecodeError as error:
        raise HumanSamplingV2Error(f"source is not UTF-8: {path}") from error


def _load_v1(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(_decode_bound_text(path, V1_PHYSICAL_SHA256))
    except json.JSONDecodeError as error:
        raise HumanSamplingV2Error("v1 artifact is not valid JSON") from error
    _require(isinstance(document, dict), "v1 artifact is not a JSON object")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == V1_SEMANTIC_SHA256
        and sha256_json(body) == V1_SEMANTIC_SHA256,
        "v1 semantic digest mismatch",
    )
    v1.verify_sampling_artifact(document)
    _require(
        document.get("sampling_recipe", {}).get("recipe_sha256") == V1_RECIPE_SHA256,
        "v1 recipe digest mismatch",
    )
    return document


def _parse_checksum_manifest(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        _require(len(line) >= 67 and line[64:66] == "  ", "malformed checksum-manifest row")
        digest = line[:64]
        reference = line[66:]
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "checksum-manifest digest is not lowercase SHA-256",
        )
        relative = Path(reference)
        _require(
            reference
            and not relative.is_absolute()
            and ".." not in relative.parts
            and not reference.startswith("./"),
            "unsafe checksum-manifest reference",
        )
        rows.append((digest, reference))
    _require(
        rows and len({reference for _, reference in rows}) == len(rows),
        "duplicate manifest path",
    )
    return rows


def _verify_checksum_manifest(path: Path, expected_sha256: str) -> list[tuple[str, str]]:
    text = _decode_bound_text(path, expected_sha256)
    rows = _parse_checksum_manifest(text)
    for expected, reference in rows:
        observed = hashlib.sha256(
            _read_regular_bytes(EVALUATION_PAPER_ROOT / reference)
        ).hexdigest()
        _require(observed == expected, f"checksum package member mismatch: {reference}")
    return rows


def _load_and_verify_sources(
    *,
    v1_path: Path,
    human_review_path: Path,
    study_yaml_path: Path,
    readiness_v1_path: Path,
    readiness_v2_path: Path,
    supersession_path: Path,
    go_package_v1_path: Path,
    go_package_v2_path: Path,
) -> dict[str, Any]:
    v1_document = _load_v1(v1_path)
    human_review = _decode_bound_text(human_review_path, HUMAN_REVIEW_PHYSICAL_SHA256)
    study_yaml = _decode_bound_text(study_yaml_path, STUDY_YAML_PHYSICAL_SHA256)
    readiness_v1 = _decode_bound_text(readiness_v1_path, READINESS_V1_PHYSICAL_SHA256)
    readiness_v2 = _decode_bound_text(readiness_v2_path, READINESS_V2_PHYSICAL_SHA256)
    supersession = _decode_bound_text(supersession_path, SUPERSESSION_PHYSICAL_SHA256)

    _require("3,072" not in human_review, "corrected human-review protocol retains 3,072")
    _require(
        "3,200 judgments: 800 unique uplift" in human_review
        and "400 concealed repeats" in human_review
        and "3,600 total" in human_review
        and "exactly 160 hours" in human_review
        and "213.333... hours" in human_review
        and "EUR 0 and USD 0" in human_review,
        "corrected human-review protocol arithmetic or boundary changed",
    )
    _require(
        "confirmatory_expert_judgment_target: 3200" in study_yaml
        and "confirmatory_expert_concealed_repeat_target: 400" in study_yaml
        and "confirmatory_expert_total_rating_presentations: 3600" in study_yaml
        and "confirmatory_expert_target_status: unfunded_not_executable" in study_yaml,
        "study YAML human target is not synchronized",
    )
    _require(
        readiness_v1.count("| 3,072 output judgments at ") == 2,
        "historical readiness v1 bytes no longer contain the recorded rows",
    )
    _require(
        readiness_v2.count("3,072") == 1
        and "two 3,072-judgment budget rows are superseded" in readiness_v2,
        "corrected readiness v2 does not confine 3,072 to historical supersession",
    )
    for required in (
        "3,200 primary judgments at 3 minutes | 160",
        "213 hours 20 minutes (213.333... h)",
        "400 repeats at 3 minutes | 20",
        "Complete 3,600-presentation output review",
        "268.8–379.2",
        "EUR 5,913.60",
        "EUR 10,428.00",
        "EUR 0/USD 0",
    ):
        _require(required in readiness_v2, f"corrected readiness v2 is missing: {required}")
    _require(
        "Status: append-only numerical correction; **NO-GO remains in force**" in supersession
        and V1_SEMANTIC_SHA256 in supersession
        and "EUR 0 and USD 0" in supersession
        and "Those bytes remain unchanged as historical evidence" in supersession,
        "append-only workload supersession boundary changed",
    )

    manifest_v1 = _verify_checksum_manifest(go_package_v1_path, GO_PACKAGE_V1_PHYSICAL_SHA256)
    manifest_v2 = _verify_checksum_manifest(go_package_v2_path, GO_PACKAGE_V2_PHYSICAL_SHA256)
    v2_references = {reference for _, reference in manifest_v2}
    required_v2_references = {
        GO_PACKAGE_V1_REFERENCE,
        HUMAN_REVIEW_REFERENCE,
        STUDY_YAML_REFERENCE,
        READINESS_V1_REFERENCE,
        READINESS_V2_REFERENCE,
        SUPERSESSION_REFERENCE,
        V1_REFERENCE,
    }
    _require(
        required_v2_references <= v2_references,
        "v2 checksum package omits a corrected or inherited source",
    )
    return {
        "v1": v1_document,
        "v1_manifest_rows": len(manifest_v1),
        "v2_manifest_rows": len(manifest_v2),
    }


def _source_commitments() -> list[dict[str, Any]]:
    return [
        {
            "role": "immutable_human_sampling_v1_superseded_not_modified",
            "reference_path": V1_REFERENCE,
            "schema_version": v1.SCHEMA_VERSION,
            "semantic_sha256": V1_SEMANTIC_SHA256,
            "physical_sha256": V1_PHYSICAL_SHA256,
        },
        {
            "role": "corrected_current_human_review_protocol",
            "reference_path": HUMAN_REVIEW_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": HUMAN_REVIEW_PHYSICAL_SHA256,
        },
        {
            "role": "corrected_current_study_yaml",
            "reference_path": STUDY_YAML_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": STUDY_YAML_PHYSICAL_SHA256,
        },
        {
            "role": "preserved_historical_readiness_v1",
            "reference_path": READINESS_V1_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": READINESS_V1_PHYSICAL_SHA256,
        },
        {
            "role": "corrected_current_readiness_v2",
            "reference_path": READINESS_V2_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": READINESS_V2_PHYSICAL_SHA256,
        },
        {
            "role": "append_only_workload_supersession_decision",
            "reference_path": SUPERSESSION_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": SUPERSESSION_PHYSICAL_SHA256,
        },
        {
            "role": "preserved_human_study_go_package_v1_checksums",
            "reference_path": GO_PACKAGE_V1_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": GO_PACKAGE_V1_PHYSICAL_SHA256,
        },
        {
            "role": "current_human_study_go_package_v2_checksums",
            "reference_path": GO_PACKAGE_V2_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": GO_PACKAGE_V2_PHYSICAL_SHA256,
        },
    ]


def _build_body(sources: Mapping[str, Any]) -> dict[str, Any]:
    v1_document = sources["v1"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "artifact_role": "append_only_documentation_bound_sampling_successor",
        "source_commitments": _source_commitments(),
        "supersession": {
            "supersedes_schema_version": v1.SCHEMA_VERSION,
            "supersedes_semantic_sha256": V1_SEMANTIC_SHA256,
            "supersedes_physical_sha256": V1_PHYSICAL_SHA256,
            "v1_bytes_modified": False,
            "sampling_coordinate_changes": 0,
            "comparison_id_changes": 0,
            "primary_judgment_slot_id_changes": 0,
            "repeat_presentation_id_changes": 0,
            "scope": (
                "bind corrected workload documentation and checksum package while preserving "
                "the exact v1 frame"
            ),
        },
        "inherited_sampling_frame": {
            "source_recipe_sha256": V1_RECIPE_SHA256,
            "outcome_blind": True,
            "arena_comparisons": 800,
            "uplift_comparisons": 800,
            "primary_judgment_slots": 3200,
            "concealed_repeat_presentations": 400,
            "total_rating_presentations": 3600,
            "balance_certificate": v1_document["balance_certificate"],
        },
        "corrected_workload_arithmetic": {
            "primary": {
                "arena_unique_comparisons": 800,
                "uplift_unique_comparisons": 800,
                "distinct_raters_per_comparison": 2,
                "judgments": 3200,
                "minutes_at_3_each": 9600,
                "hours_at_3_each": "160",
                "minutes_at_4_each": 12800,
                "hours_at_4_each": "640/3 (213.333...)",
            },
            "reliability": {
                "repeat_presentations": 400,
                "minutes_at_3_each": 1200,
                "hours_at_3_each": "20",
                "minutes_at_4_each": 1600,
                "hours_at_4_each": "80/3 (26.666...)",
            },
            "complete_output_review": {
                "presentations": 3600,
                "hours_at_3_each": "180",
                "hours_at_4_each": "240",
            },
            "validation_plus_complete_output_hours": {"minimum": "224", "maximum": "316"},
            "with_20_percent_paid_time_hours": {
                "minimum": "268.8",
                "maximum": "379.2",
            },
            "participant_earnings_eur": {
                "at_20_per_hour_with_20_percent_allowance": {
                    "minimum": "5376.00",
                    "maximum": "7584.00",
                },
                "at_25_per_hour_with_20_percent_allowance": {
                    "minimum": "6720.00",
                    "maximum": "9480.00",
                },
            },
            "illustrative_envelope_after_separate_10_percent_operational_reserve_eur": {
                "minimum": "5913.60",
                "maximum": "10428.00",
                "authorized": False,
            },
        },
        "checksum_verification": {
            "v1_package_preserved_and_verified": True,
            "v1_package_rows": sources["v1_manifest_rows"],
            "v2_package_verified": True,
            "v2_package_rows": sources["v2_manifest_rows"],
        },
        "validation_status": {
            "source_and_checksum_binding_validated": True,
            "sampling_coordinates_revalidated_via_v1": True,
            "documentation_arithmetic_validated": True,
            "power_validated": False,
            "precision_validated": False,
            "type_i_error_validated": False,
            "missingness_validated": False,
            "cost_validated": False,
            "ethics_approved": False,
            "funding_approved": False,
        },
        "claim_boundary": {
            "activation_effect": "none",
            "official": False,
            "rank_eligible": False,
            "calls_authorized": False,
            "model_calls_authorized": False,
            "epicure_calls_authorized": False,
            "human_contact_authorized": False,
            "human_judgment_collection_authorized": False,
            "compensation_or_spend_authorized": False,
            "quality_evidence_observed": False,
            "quality_observations": 0,
            "human_judgments": 0,
            "reviewer_identities_assigned": 0,
            "research_result": False,
            "paper_or_public_claim_authorized": False,
        },
    }


def build_sampling_artifact_v2(
    *,
    v1_path: Path = DEFAULT_V1_ARTIFACT,
    human_review_path: Path = DEFAULT_HUMAN_REVIEW,
    study_yaml_path: Path = DEFAULT_STUDY_YAML,
    readiness_v1_path: Path = DEFAULT_READINESS_V1,
    readiness_v2_path: Path = DEFAULT_READINESS_V2,
    supersession_path: Path = DEFAULT_SUPERSESSION,
    go_package_v1_path: Path = DEFAULT_GO_PACKAGE_V1,
    go_package_v2_path: Path = DEFAULT_GO_PACKAGE_V2,
) -> dict[str, Any]:
    """Build a deterministic v2 document from exact corrected local sources."""

    sources = _load_and_verify_sources(
        v1_path=v1_path,
        human_review_path=human_review_path,
        study_yaml_path=study_yaml_path,
        readiness_v1_path=readiness_v1_path,
        readiness_v2_path=readiness_v2_path,
        supersession_path=supersession_path,
        go_package_v1_path=go_package_v1_path,
        go_package_v2_path=go_package_v2_path,
    )
    body = _build_body(sources)
    return {**body, "artifact_sha256": sha256_json(body)}


def verify_sampling_artifact_v2(
    document: Mapping[str, Any],
    *,
    v1_path: Path = DEFAULT_V1_ARTIFACT,
    human_review_path: Path = DEFAULT_HUMAN_REVIEW,
    study_yaml_path: Path = DEFAULT_STUDY_YAML,
    readiness_v1_path: Path = DEFAULT_READINESS_V1,
    readiness_v2_path: Path = DEFAULT_READINESS_V2,
    supersession_path: Path = DEFAULT_SUPERSESSION,
    go_package_v1_path: Path = DEFAULT_GO_PACKAGE_V1,
    go_package_v2_path: Path = DEFAULT_GO_PACKAGE_V2,
) -> None:
    """Fail unless v2 exactly matches v1 and all corrected source bytes."""

    _require(isinstance(document, Mapping), "v2 artifact must be a JSON object")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == sha256_json(body),
        "v2 semantic digest mismatch",
    )
    expected = build_sampling_artifact_v2(
        v1_path=v1_path,
        human_review_path=human_review_path,
        study_yaml_path=study_yaml_path,
        readiness_v1_path=readiness_v1_path,
        readiness_v2_path=readiness_v2_path,
        supersession_path=supersession_path,
        go_package_v1_path=go_package_v1_path,
        go_package_v2_path=go_package_v2_path,
    )
    _require(document == expected, "v2 artifact differs from exact corrected successor")


def materialize_sampling_frame_v2(
    document: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Materialize the unchanged v1 frame after verifying all v2 bindings."""

    verify_sampling_artifact_v2(document)
    v1_document = _load_v1(DEFAULT_V1_ARTIFACT)
    return v1.materialize_sampling_frame(v1_document)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_sampling_artifact_v2(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Publish v2 atomically without ever replacing an existing final path."""

    verify_sampling_artifact_v2(document)
    _require(not output_dir.is_symlink(), "output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"human-judgment-sampling-v2-candidate-{digest}.json"
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists() or destination.is_symlink():
        existing = _read_regular_bytes(destination)
        _require(existing == rendered.encode(), "existing final artifact conflicts")
        return destination

    descriptor, temporary_name = tempfile.mkstemp(prefix=".human-sampling-v2-", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise HumanSamplingV2Error(
                "final artifact appeared during no-replace publication"
            ) from error
        _fsync_directory(output_dir)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-candidate", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_sampling_artifact_v2()
    verify_sampling_artifact_v2(document)
    if args.write_candidate:
        print(write_sampling_artifact_v2(document, args.output_dir))
    else:
        print(document["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
