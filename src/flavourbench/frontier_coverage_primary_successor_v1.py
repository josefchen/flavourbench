"""Freeze the corrected, development-only v4 successor to coverage plan f798.

Every operation in this module is offline.  The successor retires every
predecessor execution identity and derives fresh cell, work, run, arm,
attempt, and batch identities from a fixed nonce and namespace.  Its sole arm
form is ``run_id:epicure_on`` and every attempt slot binds that exact value.
It also binds the exact route-manifest records, but does not attest live route
availability and does not authorize execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .frontier_contract_runner import IntegrityError, load_candidate_manifest, select_candidates
from .frontier_coverage_primary_source_closure_v1 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)
from .real_dataset_runner import load_development_task_inventory
from .real_task_bank import sha256_json

PLAN_SCHEMA = "flavourbench-frontier-coverage-primary-successor-plan-v4-r1"
PREFLIGHT_SCHEMA = "flavourbench-frontier-coverage-primary-successor-preflight-v4-r1"
DRY_RUN_SCHEMA = "flavourbench-frontier-coverage-primary-successor-dry-run-v4-r1"
COHERE_OPERATOR_ATTESTATION_SCHEMA = "flavourbench-cohere-scholars-operator-attestation-v1"
COHERE_OPERATOR_TEMPLATE_SCHEMA = (
    "flavourbench-cohere-scholars-operator-attestation-template-v4-r1"
)
FREEZE_NONCE = "frontier-coverage-primary-on-v5-successor-v4-r1-2026-08-08"
NAMESPACE = uuid.UUID("36781a4b-4441-4058-8315-5f866218eb59")
NON_USD_UNKNOWN_STATUS = "unpriced_non_usd_unknown_requires_operator_attestation"
DEFAULT_OUTPUT_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v4-r1"
)

RETIRED_V4_PLAN_SHA256 = "92b6509d27e0e40d0c46573da1cbe899fc8efb34b3d541fe6b78e559411a6838"
RETIRED_V4_PREFLIGHT_SHA256 = "53f7797f1ec9a54b7be1185e1c0cc60617af8d7d9e298f0c86d986ef15d02896"
RETIRED_V4_DRY_RUN_SHA256 = "5fb69e6624cf7a1b263d5b26aa54f2116243a9e0f0d490f2079da080b2bed6ec"
RETIRED_V4_TEMPLATE_SHA256 = "52245c814d5fac890b36885f5c685b51934e9db50f0c1b7b4d5f22784c4004d0"
RETIRED_V4_RECEIPT_SHA256 = "ff30ae1a769e97868b7c62d386d9dba8b0f5c2e8f8407c8e32366ba7baa3123e"
RETIRED_V4_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v4"
)
RETIRED_V4_ALTERNATE_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v4-determinism-check"
)
RETIRED_V4_RECEIPT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v4-same-operator-technical-review-v1/"
    "frontier-coverage-primary-successor-v4-same-operator-technical-review-"
    f"{RETIRED_V4_RECEIPT_SHA256}.json"
)
RETIRED_V4_ARTIFACTS = (
    (
        "plan",
        "plan/frontier-coverage-primary-successor-plan-" f"{RETIRED_V4_PLAN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-plan-v4",
        RETIRED_V4_PLAN_SHA256,
        "03b8b690d90f39f5d0ac9fad3d5f6e7cb033edfbfcb905e1471a7e3ba3e7de2b",
    ),
    (
        "preflight",
        "preflight/frontier-coverage-primary-successor-preflight-"
        f"{RETIRED_V4_PREFLIGHT_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-preflight-v4",
        RETIRED_V4_PREFLIGHT_SHA256,
        "9a9b3ec38f49d2733733765e34398ab977b13f2f65bf5b575c9813c25f05d2a8",
    ),
    (
        "dry_run",
        "dry-run/frontier-coverage-primary-successor-dry-run-"
        f"{RETIRED_V4_DRY_RUN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-dry-run-v4",
        RETIRED_V4_DRY_RUN_SHA256,
        "d66e02bcae696da289fbc093fe48c90ed70d85e15593803e5f4d72fb0a7f3f89",
    ),
    (
        "cohere_operator_template",
        "templates/cohere-scholars-operator-attestation-template-"
        f"{RETIRED_V4_TEMPLATE_SHA256}.json",
        "flavourbench-cohere-scholars-operator-attestation-template-v4",
        RETIRED_V4_TEMPLATE_SHA256,
        "582849dd5646ee8592580a93c5419709ead35b91194a6af96312d7c71904ce2e",
    ),
)

FAILED_V1_PLAN_SHA256 = "0dc2a6ec2051563541066866bc5c6659755cd24f1c754db068d75a89ce5a4d71"
FAILED_V1_PREFLIGHT_SHA256 = "ef102dd3b7ee4ab571b80ed1be63b08f0ec188395b661f5666ad9a41da32cd1e"
FAILED_V1_DRY_RUN_SHA256 = "edc5dc3086a803c57a89fa895092501e13970b71febf547ec49016fbb23d26da"
FAILED_V1_COHERE_TEMPLATE_SHA256 = (
    "c8b73230355087e71fc89030cc82b8ada2eb7a0717920cbaa073c8cf3b5e529b"
)
FAILED_V1_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v1"
)
FAILED_V1_ARTIFACTS = (
    (
        "plan",
        f"{FAILED_V1_ROOT}/plan/frontier-coverage-primary-successor-plan-"
        f"{FAILED_V1_PLAN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-plan-v1",
        FAILED_V1_PLAN_SHA256,
        "06185a33822a8a7226cda95a0262aa41dd986abbbc4f4e5d936ec15856322213",
    ),
    (
        "preflight",
        f"{FAILED_V1_ROOT}/preflight/frontier-coverage-primary-successor-preflight-"
        f"{FAILED_V1_PREFLIGHT_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-preflight-v1",
        FAILED_V1_PREFLIGHT_SHA256,
        "3690067e4884c976a26dfce8c2274ea50fa04826d8ed013bc72009429f5601d8",
    ),
    (
        "dry_run",
        f"{FAILED_V1_ROOT}/dry-run/frontier-coverage-primary-successor-dry-run-"
        f"{FAILED_V1_DRY_RUN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-dry-run-v1",
        FAILED_V1_DRY_RUN_SHA256,
        "07e027c3908ced782d977b575c82b227f1120a23657d5aa13006a17ffcc0fb76",
    ),
    (
        "cohere_operator_template",
        f"{FAILED_V1_ROOT}/templates/cohere-scholars-operator-attestation-template-"
        f"{FAILED_V1_COHERE_TEMPLATE_SHA256}.json",
        "flavourbench-cohere-scholars-operator-attestation-template-v1",
        FAILED_V1_COHERE_TEMPLATE_SHA256,
        "42a35272d92e37d209e7134d4e43941621cbb8c04eaf11edc8c31e0489e18b60",
    ),
)

FAILED_V2_PLAN_SHA256 = "e04285b32d9f2543bb9f8c803762e6621c2674e4120313236044ad80f62b9ff7"
FAILED_V2_PREFLIGHT_SHA256 = "9fc35dac5b0be998e55abb836c4e724ae645d707f2430b45ebbd411854b5a955"
FAILED_V2_DRY_RUN_SHA256 = "9979bcbf758b46076310ece99e798c2337df7408560d16a4428a7134dbad60fd"
FAILED_V2_COHERE_TEMPLATE_SHA256 = (
    "7911915106c9eb7368fad531943ba941bb93853fa46bca574e8cc575fb1013d2"
)
FAILED_V2_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v2"
)
FAILED_V2_ARTIFACTS = (
    (
        "plan",
        f"{FAILED_V2_ROOT}/plan/frontier-coverage-primary-successor-plan-"
        f"{FAILED_V2_PLAN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-plan-v2",
        FAILED_V2_PLAN_SHA256,
        "155d3a5a591f4f958c9b00799d8396eb0f0402bf8db52dfc2b5cdecba4076f4a",
    ),
    (
        "preflight",
        f"{FAILED_V2_ROOT}/preflight/frontier-coverage-primary-successor-preflight-"
        f"{FAILED_V2_PREFLIGHT_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-preflight-v2",
        FAILED_V2_PREFLIGHT_SHA256,
        "1c41b80913e1d8b34573ab8bc93d2e2b26909efb8d0f7db4a0cc4eb53d3f188a",
    ),
    (
        "dry_run",
        f"{FAILED_V2_ROOT}/dry-run/frontier-coverage-primary-successor-dry-run-"
        f"{FAILED_V2_DRY_RUN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-dry-run-v2",
        FAILED_V2_DRY_RUN_SHA256,
        "20ad9481dce62f858bb6ed6ccc6a4cb9efaa7a2c022f65f161ce0ffff965c46f",
    ),
    (
        "cohere_operator_template",
        f"{FAILED_V2_ROOT}/templates/cohere-scholars-operator-attestation-template-"
        f"{FAILED_V2_COHERE_TEMPLATE_SHA256}.json",
        "flavourbench-cohere-scholars-operator-attestation-template-v2",
        FAILED_V2_COHERE_TEMPLATE_SHA256,
        "51664944f390bdde87905209c8b6003f37b59f0170e78ca4cb6c8c839be03b1f",
    ),
)

FAILED_V3_PLAN_SHA256 = "27379802f6893aebbcbfb2dcf3beb70352b7ae414048084740821d9f258d5cd5"
FAILED_V3_PREFLIGHT_SHA256 = "b939fc9993a664d36b8519a8edf3a3fd7d417b26c79a4f4647b03a0c8e9d2b75"
FAILED_V3_DRY_RUN_SHA256 = "33b7ee123b5a468fc7e623fa72ffbc94511012ea84c1433d8ca3182375805024"
FAILED_V3_COHERE_TEMPLATE_SHA256 = (
    "2d5f04176341b64313e8ec7f4b0335540e7210a6b903bf74aebdd8988022b908"
)
FAILED_V3_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v3"
)
FAILED_V3_ARTIFACTS = (
    (
        "plan",
        f"{FAILED_V3_ROOT}/plan/frontier-coverage-primary-successor-plan-"
        f"{FAILED_V3_PLAN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-plan-v3",
        FAILED_V3_PLAN_SHA256,
        "3a02b2af4e66823a8a54f212dcd0e170fca254aaf84eb0b6af9850e0d46fb0c0",
    ),
    (
        "preflight",
        f"{FAILED_V3_ROOT}/preflight/frontier-coverage-primary-successor-preflight-"
        f"{FAILED_V3_PREFLIGHT_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-preflight-v3",
        FAILED_V3_PREFLIGHT_SHA256,
        "5d03ddb1184b92c22195953b963db73ca1ed353e9f7ba761a638d685ee159c3d",
    ),
    (
        "dry_run",
        f"{FAILED_V3_ROOT}/dry-run/frontier-coverage-primary-successor-dry-run-"
        f"{FAILED_V3_DRY_RUN_SHA256}.json",
        "flavourbench-frontier-coverage-primary-successor-dry-run-v3",
        FAILED_V3_DRY_RUN_SHA256,
        "ee9f8f6ef1485e79562cbb700674803c9b595a72c702884f442f0379243147d0",
    ),
    (
        "cohere_operator_template",
        f"{FAILED_V3_ROOT}/templates/cohere-scholars-operator-attestation-template-"
        f"{FAILED_V3_COHERE_TEMPLATE_SHA256}.json",
        "flavourbench-cohere-scholars-operator-attestation-template-v3",
        FAILED_V3_COHERE_TEMPLATE_SHA256,
        "bba75a68d220146614c73b301600438ff6acdb1d40c63451af111d19e3e14788",
    ),
)
FAILED_V3_AUDIT_SHA256 = "681c3e068726170d5508162eff55f16fc7f09ac0967478fb083dd466b7122078"
FAILED_V3_AUDIT_FILE_SHA256 = (
    "369752f5aff0d8035c8863dc1f5cf89fea775bab238f617c8141bc49d177bfe8"
)
FAILED_V3_AUDIT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "frontier-coverage-primary-on-v5-successor-v3-independent-audit-v1/"
    "frontier-coverage-primary-successor-v3-independent-audit-"
    f"{FAILED_V3_AUDIT_SHA256}.json"
)

PREDECESSOR_SHA256 = "f79850aaa6a9b256340c2932ae376e6887e387b7bded6ce2ffd06d7caa3dc308"
PREDECESSOR = (
    "flavourbench/artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/"
    f"frontier-coverage-primary-on-v5-plan-{PREDECESSOR_SHA256}.json"
)
PREDECESSOR_PREFLIGHT_SHA256 = "4b0be120e32f5f8e448742a1411ed48cccf64f0af29c359a28cf0f6a1eaa1797"
PREDECESSOR_PREFLIGHT = (
    "flavourbench/artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/"
    "preflight/frontier-coverage-primary-preflight-"
    f"{PREDECESSOR_PREFLIGHT_SHA256}.json"
)
COHERE_GATE_SHA256 = "a89c319e32ba169645173809b1019a51b549dfdc22cab75f06c4d5718cb8f918"
COHERE_GATE = (
    "flavourbench/artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/"
    "route-gate/frontier-coverage-primary-cohere-route-gate-"
    f"{COHERE_GATE_SHA256}.json"
)
CORRECTED_ARENA_SHA256 = "234f5b5e3364f0e0f2fddc0f23d47d1d670df509c5707e35cb713183264c5c5e"
CORRECTED_ARENA = (
    "flavourbench/artifacts/season1/current-quality-run/frontier-coverage-v4-postrun/"
    f"frontier-corrected-development-arena-{CORRECTED_ARENA_SHA256}.json"
)
TASK_VALIDITY_SHA256 = "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119"
TASK_VALIDITY = (
    "flavourbench/artifacts/season1/task-validity/development-v2/"
    f"development-task-validity-v2-{TASK_VALIDITY_SHA256}.json"
)
TASK_QUARANTINE_SHA256 = "e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15"
TASK_QUARANTINE = (
    "flavourbench/artifacts/season1/current-quality-run/task-quarantine-v1/"
    f"current-frontier-task-quarantine-{TASK_QUARANTINE_SHA256}.json"
)
ROUTE_MANIFESTS = (
    "flavourbench/artifacts/season1/current-quality-run/manifest-v29-high-resource/"
    "flavourbench-routed-unranked-"
    "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json",
    "flavourbench/artifacts/season1/current-quality-run/manifest-v42-high-resource-cohere-direct/"
    "flavourbench-cohere-unranked-"
    "fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json",
)

CURRENT_EXPOSURE_USD = Decimal("48.01944682666666666666666666")
ADMISSION_CEILING_USD = Decimal("85")
HARD_CAP_USD = Decimal("100")
HISTORICAL_UNPRICED_COHERE_WORK_IDS = (
    "63cf4b5c57e627ae17d150c6d0a37d30b7f59bee1c1f9a301a6c48c30b700a79",
    "5b96a44e626eb9005c0ace07e7ac6378b2eb5e33eae22988a2cbf38080c16d6c",
    "f76df88cbf4f0b179c34b741513c59f8014734a54037126bd7a7b761545a2427",
)
HISTORICAL_UNPRICED_COHERE_LEDGERS = {
    HISTORICAL_UNPRICED_COHERE_WORK_IDS[0]: (
        "flavourbench/artifacts/season1/current-quality-run/"
        "frontier-coverage-repair-execution-v1/ledger.jsonl"
    ),
    HISTORICAL_UNPRICED_COHERE_WORK_IDS[1]: (
        "flavourbench/artifacts/season1/current-quality-run/pilot-v34-cohere-direct/ledger.jsonl"
    ),
    HISTORICAL_UNPRICED_COHERE_WORK_IDS[2]: (
        "flavourbench/artifacts/season1/current-quality-run/pilot-v37-cohere-direct/ledger.jsonl"
    ),
}


class CoverageSuccessorError(RuntimeError):
    """A frozen coverage identity, route, or accounting predicate failed."""


def _historical_cohere_disclosure(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for work_id in HISTORICAL_UNPRICED_COHERE_WORK_IDS:
        path = repo_root / HISTORICAL_UNPRICED_COHERE_LEDGERS[work_id]
        if path.is_symlink() or not path.is_file():
            raise CoverageSuccessorError(f"historical Cohere ledger is absent: {path}")
        entries: list[dict[str, Any]] = []
        previous: str | None = None
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise CoverageSuccessorError("historical Cohere ledger has a blank line")
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise CoverageSuccessorError("historical Cohere ledger is invalid JSONL") from error
            if (
                not isinstance(entry, dict)
                or entry.get("schema_version") != "flavourbench-real-exploratory-ledger-v1"
                or entry.get("sequence") != number
                or entry.get("previous_entry_sha256") != previous
                or entry.get("entry_sha256")
                != _sha256({key: value for key, value in entry.items() if key != "entry_sha256"})
            ):
                raise CoverageSuccessorError("historical Cohere ledger hash chain failed")
            entries.append(entry)
            previous = str(entry["entry_sha256"])
        reservations = [
            entry
            for entry in entries
            if entry.get("event_type") == "reservation_created"
            and entry.get("work_item_id") == work_id
        ]
        finalizations = [
            entry
            for entry in entries
            if entry.get("event_type") == "source_artifact_recorded"
            and entry.get("work_item_id") == work_id
        ]
        incidents = [
            entry
            for entry in entries
            if entry.get("event_type") == "execution_incident"
            and entry.get("work_item_id") == work_id
        ]
        if (
            len(reservations) != 1
            or reservations[0].get("reserved_usd") != "0"
            or reservations[0].get("provider_tag") != "cohere-direct"
            or not str(reservations[0].get("model_id") or "").startswith("cohere/")
            or finalizations
            or not incidents
            or any(
                incident.get("reservation_entry_sha256") != reservations[0]["entry_sha256"]
                for incident in incidents
            )
        ):
            raise CoverageSuccessorError("historical Cohere zero-reservation evidence differs")
        records.append(
            {
                "work_item_id": work_id,
                "model_id": reservations[0]["model_id"],
                "provider_tag": "cohere-direct",
                "reserved_usd_recorded": "0",
                "usd_cost_interpretation": "unknown_unpriced_not_free",
                "source_finalization_present": False,
                "incident_entry_sha256s": [entry["entry_sha256"] for entry in incidents],
                "successor_replay_permitted": False,
                "ledger": {
                    "path": _relative(repo_root, path),
                    "bytes": path.stat().st_size,
                    "file_sha256": _file_sha256(path),
                    "schema_version": "flavourbench-real-exploratory-ledger-v1",
                    "entry_count": len(entries),
                    "head_entry_sha256": entries[-1]["entry_sha256"],
                },
            }
        )
    return {
        "status": "closed_no_replay_unpriced_unknown_cost_disclosed",
        "usd_exposure_claimed": False,
        "blocks_priced_openrouter_or_kimi_batches": False,
        "records": records,
        "records_sha256": sha256_json(records),
    }


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise CoverageSuccessorError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise CoverageSuccessorError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _contains_secret_material(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if (
                lowered
                in {
                    "contains_secret",
                    "credential_binding_is_derived_from_secret",
                }
                and child is False
            ):
                continue
            if (
                "api_key" in lowered
                or "secret_hash" in lowered
                or "secret_sha" in lowered
                or "credential_fingerprint" in lowered
                or lowered in {"credential_sha256", "key_fingerprint"}
                or lowered
                in {"authorization", "credential", "credentials", "password", "secret", "token"}
            ):
                return True
            if _contains_secret_material(child):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        return bool(
            re.search(r"(?:sk-[A-Za-z0-9_-]{16,}|cohere_[A-Za-z0-9._-]{16,})", value)
            or re.search(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", value, re.IGNORECASE)
            or re.search(r"\bAKIA[0-9A-Z]{16}\b", value)
        )
    return False


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CoverageSuccessorError(f"expected a regular non-symlink JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageSuccessorError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise CoverageSuccessorError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CoverageSuccessorError(f"expected a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise CoverageSuccessorError(f"path escapes repository: {path}") from error


def _addressed(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    document = _regular_json(path)
    digest = document.get("artifact_sha256") or (document.get("content_address") or {}).get(
        "digest"
    )
    if not isinstance(digest, str) or len(digest) != 64:
        raise CoverageSuccessorError(f"artifact has no semantic SHA-256: {path}")
    if "artifact_sha256" in document:
        body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    else:
        body = {key: value for key, value in document.items() if key != "content_address"}
    if sha256_json(body) != digest or (expected_sha256 and digest != expected_sha256):
        raise CoverageSuccessorError(f"content address does not verify: {path}")
    return document


def _file_ref(repo_root: Path, path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    document = _addressed(path, expected_sha256=expected_sha256)
    semantic = document.get("artifact_sha256") or document["content_address"]["digest"]
    return {
        "path": _relative(repo_root, path),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "semantic_sha256": semantic,
    }


def _plan_execution_identifiers(plan: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for cell in plan.get("cells") or []:
        if not isinstance(cell, Mapping):
            raise CoverageSuccessorError("plan contains a non-object cell")
        identifiers.update(
            str(value)
            for value in (cell.get("cell_id"), cell.get("work_item_id"), cell.get("run_id"))
            if isinstance(value, str) and value
        )
        arm_ids = cell.get("arm_ids") or {}
        if not isinstance(arm_ids, Mapping):
            raise CoverageSuccessorError("plan contains malformed arm IDs")
        identifiers.update(
            str(value) for value in arm_ids.values() if isinstance(value, str) and value
        )
        for slot in cell.get("attempt_slots") or []:
            if not isinstance(slot, Mapping):
                raise CoverageSuccessorError("plan contains a non-object attempt slot")
            identifiers.update(
                str(value)
                for value in (slot.get("attempt_id"), slot.get("arm_id"))
                if isinstance(value, str) and value
            )
    for batch in plan.get("endpoint_batches") or []:
        if not isinstance(batch, Mapping):
            raise CoverageSuccessorError("plan contains a non-object endpoint batch")
        batch_id = batch.get("batch_id")
        if isinstance(batch_id, str) and batch_id:
            identifiers.add(batch_id)
    identifiers.discard("")
    return identifiers


def _ordered_cohere_work_item_ids(plan: Mapping[str, Any]) -> list[str]:
    return [
        str(cell["work_item_id"])
        for cell in plan.get("cells") or []
        if isinstance(cell, Mapping) and cell.get("execution_backend") == "cohere_direct"
    ]


def _current_cohere_economic_ambiguities(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    """Recursively reject generic current zero/free Cohere economics.

    Literal source labels are allowed only below ``historical_source_provenance``.
    That subtree is an immutable disclosure, not a current route, budget, price, or
    reservation representation.  Boolean denials are not numeric zero claims.
    """

    economic_fragments = (
        "price",
        "pricing",
        "cost",
        "reserve",
        "reservation",
        "usd",
        "currency",
        "rate",
    )
    findings: list[dict[str, str]] = []

    def visit(
        value: object, *, path: str, key: str = "", economic_context: bool = False
    ) -> None:
        if key == "historical_source_provenance":
            return
        economic = economic_context or any(
            fragment in key.lower() for fragment in economic_fragments
        )
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(
                    child,
                    path=f"{path}.{child_key}",
                    key=str(child_key),
                    economic_context=economic,
                )
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=f"{path}[{index}]",
                    key=key,
                    economic_context=economic,
                )
            return
        if not economic or isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float, Decimal)) and Decimal(str(value)) == 0:
            findings.append({"path": path, "reason": "generic_numeric_zero"})
            return
        if isinstance(value, str):
            lowered = value.strip().lower()
            try:
                numeric_zero = Decimal(lowered) == 0
            except InvalidOperation:
                numeric_zero = False
            if numeric_zero:
                findings.append({"path": path, "reason": "generic_numeric_zero"})
            elif "free" in lowered:
                findings.append({"path": path, "reason": "generic_free_label"})

    for index, cell in enumerate(plan.get("cells") or []):
        if isinstance(cell, Mapping) and cell.get("execution_backend") == "cohere_direct":
            visit(cell, path=f"cells[{index}]")
    for index, batch in enumerate(plan.get("endpoint_batches") or []):
        if isinstance(batch, Mapping) and batch.get("execution_backend") == "cohere_direct":
            visit(batch, path=f"endpoint_batches[{index}]")
    return sorted(findings, key=lambda row: (row["path"], row["reason"]))


def _failed_v1_freeze_evidence(repo_root: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for role, relative, schema, semantic_sha256, physical_sha256 in FAILED_V1_ARTIFACTS:
        path = repo_root / relative
        document = _addressed(path, expected_sha256=semantic_sha256)
        if document.get("schema_version") != schema or _file_sha256(path) != physical_sha256:
            raise CoverageSuccessorError(f"failed v1 {role} artifact identity differs")
        documents[role] = document
        references.append(
            {"role": role, **_file_ref(repo_root, path, expected_sha256=semantic_sha256)}
        )

    plan = documents["plan"]
    preflight = documents["preflight"]
    dry_run = documents["dry_run"]
    template = documents["cohere_operator_template"]
    plan_counts = plan.get("counts") or {}
    dry_counts = dry_run.get("counts") or {}
    zero_calls = {
        "plan": {
            "provider_completions": plan_counts.get("provider_calls_by_freeze"),
            "catalog_gets": plan_counts.get("catalog_calls_by_freeze"),
            "epicure": plan_counts.get("epicure_calls_by_freeze"),
        },
        "preflight": dict(preflight.get("calls_made") or {}),
        "dry_run": {
            "provider_completions": dry_counts.get("provider_completions"),
            "catalog_gets": dry_counts.get("catalog_gets"),
            "epicure": dry_counts.get("epicure_calls"),
        },
    }
    expected_zero = {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
    retired = _plan_execution_identifiers(plan)
    if (
        plan.get("artifact_sha256") != FAILED_V1_PLAN_SHA256
        or preflight.get("plan_sha256") != FAILED_V1_PLAN_SHA256
        or preflight.get("decision") != "execution_not_admitted"
        or dry_run.get("plan_sha256") != FAILED_V1_PLAN_SHA256
        or dry_run.get("preflight_sha256") != FAILED_V1_PREFLIGHT_SHA256
        or template.get("plan_sha256") != FAILED_V1_PLAN_SHA256
        or any(counters != expected_zero for counters in zero_calls.values())
        or len(retired) != 1666
    ):
        raise CoverageSuccessorError("failed v1 four-artifact incident chain differs")
    return {
        "schema_version": "flavourbench-frontier-coverage-failed-offline-freeze-evidence-v1",
        "status": "failed_offline_freeze_retired_zero_calls",
        "supersedes_failed_offline_plan_sha256": FAILED_V1_PLAN_SHA256,
        "failure_reason": "second_exact_real_root_freeze_rejected_owned_output",
        "artifacts": references,
        "zero_call_counters": zero_calls,
        "execution_admission_granted": False,
        "provider_or_epicure_calls_made": False,
        "retired_v1_identifier_count": len(retired),
        "retired_v1_identifiers_sha256": sha256_json(sorted(retired)),
        "retired_v1_identifiers_replay_permitted": False,
    }


def _failed_v2_freeze_evidence(repo_root: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for role, relative, schema, semantic_sha256, physical_sha256 in FAILED_V2_ARTIFACTS:
        path = repo_root / relative
        document = _addressed(path, expected_sha256=semantic_sha256)
        if document.get("schema_version") != schema or _file_sha256(path) != physical_sha256:
            raise CoverageSuccessorError(f"failed v2 {role} artifact identity differs")
        documents[role] = document
        references.append(
            {"role": role, **_file_ref(repo_root, path, expected_sha256=semantic_sha256)}
        )

    plan = documents["plan"]
    preflight = documents["preflight"]
    dry_run = documents["dry_run"]
    template = documents["cohere_operator_template"]
    plan_counts = plan.get("counts") or {}
    dry_counts = dry_run.get("counts") or {}
    zero_calls = {
        "plan": {
            "provider_completions": plan_counts.get("provider_calls_by_freeze"),
            "catalog_gets": plan_counts.get("catalog_calls_by_freeze"),
            "epicure": plan_counts.get("epicure_calls_by_freeze"),
        },
        "preflight": dict(preflight.get("calls_made") or {}),
        "dry_run": {
            "provider_completions": dry_counts.get("provider_completions"),
            "catalog_gets": dry_counts.get("catalog_gets"),
            "epicure": dry_counts.get("epicure_calls"),
        },
    }
    expected_zero = {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
    retired = _plan_execution_identifiers(plan)
    cohere_cells = [
        cell
        for cell in plan.get("cells") or []
        if isinstance(cell, Mapping) and cell.get("execution_backend") == "cohere_direct"
    ]
    cohere_batches = [
        batch
        for batch in plan.get("endpoint_batches") or []
        if isinstance(batch, Mapping) and batch.get("execution_backend") == "cohere_direct"
    ]
    v1_evidence = _failed_v1_freeze_evidence(repo_root)
    if (
        plan.get("artifact_sha256") != FAILED_V2_PLAN_SHA256
        or plan.get("supersedes_failed_offline_plan_sha256") != FAILED_V1_PLAN_SHA256
        or plan.get("failed_offline_supersession") != v1_evidence
        or preflight.get("plan_sha256") != FAILED_V2_PLAN_SHA256
        or preflight.get("decision") != "execution_not_admitted"
        or dry_run.get("plan_sha256") != FAILED_V2_PLAN_SHA256
        or dry_run.get("preflight_sha256") != FAILED_V2_PREFLIGHT_SHA256
        or template.get("plan_sha256") != FAILED_V2_PLAN_SHA256
        or any(counters != expected_zero for counters in zero_calls.values())
        or len(retired) != 1666
        or len(cohere_cells) != 8
        or any(
            (cell.get("cost_reservation") or {}).get("status")
            != "non_usd_resource_governed_requires_envelope_and_operator_attestation"
            or (cell.get("cost_reservation") or {}).get("successor_reservation_usd") != "0"
            or (cell.get("cost_reservation") or {}).get("zero_is_free_claimed") is not False
            for cell in cohere_cells
        )
        or len(cohere_batches) != 2
        or any(
            batch.get("complete_reservation_bound") is not True
            or batch.get("unpriced_cell_count") != 0
            or batch.get("successor_priced_reserve_usd") != "0"
            for batch in cohere_batches
        )
    ):
        raise CoverageSuccessorError("failed v2 four-artifact incident chain differs")
    return {
        "schema_version": "flavourbench-frontier-coverage-failed-offline-freeze-evidence-v2",
        "status": "failed_offline_freeze_retired_zero_calls",
        "supersedes_failed_offline_plan_sha256": FAILED_V2_PLAN_SHA256,
        "failure_reason": "cohere_non_usd_cells_misrepresented_as_zero_usd_reservations",
        "artifacts": references,
        "zero_call_counters": zero_calls,
        "execution_admission_granted": False,
        "provider_or_epicure_calls_made": False,
        "failed_v1_evidence_sha256": sha256_json(v1_evidence),
        "retired_v2_identifier_count": len(retired),
        "retired_v2_identifiers_sha256": sha256_json(sorted(retired)),
        "retired_v2_identifiers_replay_permitted": False,
        "cohere_cells_with_zero_usd_misrepresentation": len(cohere_cells),
        "cohere_batches_with_false_complete_reservation_claim": len(cohere_batches),
    }


def _failed_v3_freeze_evidence(repo_root: Path) -> dict[str, Any]:
    """Bind the exact V3 freeze and the independent audit that rejected it.

    V3 is evidence, never an executable predecessor.  Both semantic and physical
    identities are checked so a re-rendered or superseding document cannot be
    substituted for the reviewed NO-GO record.
    """

    documents: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for role, relative, schema, semantic_sha256, physical_sha256 in FAILED_V3_ARTIFACTS:
        path = repo_root / relative
        document = _addressed(path, expected_sha256=semantic_sha256)
        if document.get("schema_version") != schema or _file_sha256(path) != physical_sha256:
            raise CoverageSuccessorError(f"failed v3 {role} artifact identity differs")
        documents[role] = document
        references.append(
            {"role": role, **_file_ref(repo_root, path, expected_sha256=semantic_sha256)}
        )

    audit_path = repo_root / FAILED_V3_AUDIT
    audit = _addressed(audit_path, expected_sha256=FAILED_V3_AUDIT_SHA256)
    if (
        audit.get("schema_version")
        != "flavourbench-frontier-coverage-primary-successor-v3-independent-audit-v1"
        or _file_sha256(audit_path) != FAILED_V3_AUDIT_FILE_SHA256
        or audit.get("decision") != "NO_GO"
    ):
        raise CoverageSuccessorError("failed v3 independent NO-GO identity differs")

    plan = documents["plan"]
    preflight = documents["preflight"]
    dry_run = documents["dry_run"]
    template = documents["cohere_operator_template"]
    v2_evidence = _failed_v2_freeze_evidence(repo_root)
    plan_counts = plan.get("counts") or {}
    dry_counts = dry_run.get("counts") or {}
    zero_calls = {
        "plan": {
            "provider_completions": plan_counts.get("provider_calls_by_freeze"),
            "catalog_gets": plan_counts.get("catalog_calls_by_freeze"),
            "epicure": plan_counts.get("epicure_calls_by_freeze"),
        },
        "preflight": dict(preflight.get("calls_made") or {}),
        "dry_run": {
            "provider_completions": dry_counts.get("provider_completions"),
            "catalog_gets": dry_counts.get("catalog_gets"),
            "epicure": dry_counts.get("epicure_calls"),
        },
    }
    expected_zero = {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
    retired = _plan_execution_identifiers(plan)
    bindings = audit.get("bindings") or {}
    expected_bindings = {
        "plan": {key: value for key, value in references[0].items() if key != "role"},
        "preflight": {key: value for key, value in references[1].items() if key != "role"},
        "dry_run": {key: value for key, value in references[2].items() if key != "role"},
        "operator_attestation_template": {
            key: value for key, value in references[3].items() if key != "role"
        },
    }
    authorization = audit.get("authorization") or {}
    calls_check = (audit.get("checks") or {}).get("calls_and_records") or {}
    cohere_check = (audit.get("checks") or {}).get("cohere_non_usd_semantics") or {}
    deterministic_check = (audit.get("checks") or {}).get("deterministic_freeze") or {}
    if (
        plan.get("artifact_sha256") != FAILED_V3_PLAN_SHA256
        or plan.get("supersedes_failed_offline_plan_sha256") != FAILED_V2_PLAN_SHA256
        or plan.get("failed_offline_supersession") != v2_evidence
        or preflight.get("plan_sha256") != FAILED_V3_PLAN_SHA256
        or preflight.get("decision") != "execution_not_admitted"
        or dry_run.get("plan_sha256") != FAILED_V3_PLAN_SHA256
        or dry_run.get("preflight_sha256") != FAILED_V3_PREFLIGHT_SHA256
        or template.get("plan_sha256") != FAILED_V3_PLAN_SHA256
        or any(counters != expected_zero for counters in zero_calls.values())
        or len(retired) != 1666
        or any(bindings.get(role) != reference for role, reference in expected_bindings.items())
        or any(
            value is not False
            for key, value in authorization.items()
            if key.endswith("authorized")
        )
        or calls_check.get("verdict") != "pass"
        or calls_check.get("provider_calls") != 0
        or calls_check.get("catalog_calls") != 0
        or calls_check.get("epicure_calls") != 0
        or calls_check.get("admission_records") != 0
        or calls_check.get("run_records") != 0
        or cohere_check.get("verdict") != "fail"
        or cohere_check.get("generic_current_zero_or_free_findings") != 48
        or cohere_check.get("finding_breakdown")
        != {
            "reserved_worst_case_usd_zero": 8,
            "route_pricing_numeric_zero_fields": 32,
            "route_pricing_public_free_status": 8,
        }
        or deterministic_check.get("verdict") != "fail"
        or deterministic_check.get("live_repo_alternate_output_refreeze_verifies") is not False
    ):
        raise CoverageSuccessorError("failed v3 independent NO-GO chain differs")
    return {
        "schema_version": (
            "flavourbench-frontier-coverage-failed-offline-freeze-evidence-v3"
        ),
        "status": "independent_no_go_retired_zero_calls",
        "supersedes_failed_offline_plan_sha256": FAILED_V3_PLAN_SHA256,
        "failure_reasons": [
            "generic_current_cohere_zero_or_free_pricing_semantics",
            "alternate_output_deterministic_refreeze_failure",
        ],
        "artifacts": references,
        "independent_audit": {
            **_file_ref(
                repo_root, audit_path, expected_sha256=FAILED_V3_AUDIT_SHA256
            ),
            "decision": "NO_GO",
        },
        "zero_call_counters": zero_calls,
        "execution_admission_granted": False,
        "provider_or_epicure_calls_made": False,
        "failed_v2_evidence_sha256": sha256_json(v2_evidence),
        "retired_v3_identifier_count": len(retired),
        "retired_v3_identifiers_sha256": sha256_json(sorted(retired)),
        "retired_v3_identifiers_replay_permitted": False,
        "generic_current_cohere_zero_or_free_findings": 48,
        "alternate_output_refreeze_verified": False,
    }


def _retired_v4_format_freeze_evidence(repo_root: Path) -> dict[str, Any]:
    """Bind the zero-call V4 freeze retired solely by a formatting refreeze."""

    roots = (RETIRED_V4_ROOT, RETIRED_V4_ALTERNATE_ROOT)
    documents_by_root: list[dict[str, dict[str, Any]]] = []
    references: dict[str, list[dict[str, Any]]] = {}
    for root in roots:
        documents: dict[str, dict[str, Any]] = {}
        root_references: list[dict[str, Any]] = []
        for role, suffix, schema, semantic_sha256, physical_sha256 in RETIRED_V4_ARTIFACTS:
            path = repo_root / root / suffix
            document = _addressed(path, expected_sha256=semantic_sha256)
            if document.get("schema_version") != schema or _file_sha256(path) != physical_sha256:
                raise CoverageSuccessorError(f"retired v4 {role} identity differs")
            documents[role] = document
            root_references.append(
                {"role": role, **_file_ref(repo_root, path, expected_sha256=semantic_sha256)}
            )
        documents_by_root.append(documents)
        references[root] = root_references
    if documents_by_root[0] != documents_by_root[1]:
        raise CoverageSuccessorError("retired v4 deterministic roots are not byte-semantic peers")

    receipt_path = repo_root / RETIRED_V4_RECEIPT
    receipt = _addressed(receipt_path, expected_sha256=RETIRED_V4_RECEIPT_SHA256)
    plan = documents_by_root[0]["plan"]
    preflight = documents_by_root[0]["preflight"]
    dry_run = documents_by_root[0]["dry_run"]
    template = documents_by_root[0]["cohere_operator_template"]
    counts = plan.get("counts") or {}
    dry_counts = dry_run.get("counts") or {}
    retired = _plan_execution_identifiers(plan)
    if (
        _file_sha256(receipt_path)
        != "6a85b54aeeb0d4a455197647ebdca45e7a8db679c46698e8baef756c29a0368e"
        or receipt.get("schema_version")
        != (
            "flavourbench-frontier-coverage-primary-successor-v4-"
            "same-operator-technical-review-v1"
        )
        or receipt.get("decision") != "TECHNICAL_PASS_INDEPENDENT_GOVERNANCE_GO_REQUIRED"
        or (receipt.get("authorization") or {}).get("run_or_execution_authorized") is not False
        or plan.get("artifact_sha256") != RETIRED_V4_PLAN_SHA256
        or plan.get("status") != "frozen_not_executed_independent_go_required"
        or plan.get("supersedes_failed_offline_plan_sha256") != FAILED_V3_PLAN_SHA256
        or (
            (plan.get("failed_offline_supersession") or {}).get("independent_audit") or {}
        ).get("semantic_sha256")
        != FAILED_V3_AUDIT_SHA256
        or preflight.get("plan_sha256") != RETIRED_V4_PLAN_SHA256
        or preflight.get("decision") != "execution_not_admitted"
        or dry_run.get("plan_sha256") != RETIRED_V4_PLAN_SHA256
        or dry_run.get("preflight_sha256") != RETIRED_V4_PREFLIGHT_SHA256
        or template.get("plan_sha256") != RETIRED_V4_PLAN_SHA256
        or counts.get("provider_calls_by_freeze") != 0
        or counts.get("catalog_calls_by_freeze") != 0
        or counts.get("epicure_calls_by_freeze") != 0
        or dry_counts.get("provider_completions") != 0
        or dry_counts.get("catalog_gets") != 0
        or dry_counts.get("epicure_calls") != 0
        or len(retired) != 1666
    ):
        raise CoverageSuccessorError("retired v4 format-only freeze evidence differs")
    return {
        "schema_version": "flavourbench-frontier-coverage-retired-format-freeze-v4-r1",
        "status": "retired_zero_call_format_only_refreeze_required",
        "retired_plan_sha256": RETIRED_V4_PLAN_SHA256,
        "retirement_reason": "ruff_e501_formatting_only_source_closure_change",
        "canonical_and_alternate_artifacts": references,
        "same_operator_technical_receipt": _file_ref(
            repo_root, receipt_path, expected_sha256=RETIRED_V4_RECEIPT_SHA256
        ),
        "provider_or_epicure_calls_made": False,
        "execution_admission_granted": False,
        "retired_v4_identifier_count": len(retired),
        "retired_v4_identifiers_sha256": sha256_json(sorted(retired)),
        "retired_v4_identifiers_replay_permitted": False,
    }


def _verify_existing_offline_output_root(
    *, output_root: Path, repo_root: Path
) -> dict[str, dict[str, Any]] | None:
    if output_root.is_symlink():
        raise CoverageSuccessorError("successor output root must not be a symlink")
    if not output_root.exists():
        return None
    if not output_root.is_dir():
        raise CoverageSuccessorError("successor output root must be a directory")
    children = sorted(output_root.iterdir(), key=lambda path: path.name)
    if not children:
        return None
    expected = {
        "plan": (
            "frontier-coverage-primary-successor-plan",
            PLAN_SCHEMA,
        ),
        "preflight": (
            "frontier-coverage-primary-successor-preflight",
            PREFLIGHT_SCHEMA,
        ),
        "dry-run": (
            "frontier-coverage-primary-successor-dry-run",
            DRY_RUN_SCHEMA,
        ),
        "templates": (
            "cohere-scholars-operator-attestation-template",
            COHERE_OPERATOR_TEMPLATE_SCHEMA,
        ),
    }
    if {path.name for path in children} != set(expected):
        raise CoverageSuccessorError(
            "successor output root contains an unverified or foreign artifact"
        )
    documents: dict[str, dict[str, Any]] = {}
    for directory in children:
        if directory.is_symlink() or not directory.is_dir():
            raise CoverageSuccessorError(
                "successor output root contains an unverified or foreign artifact"
            )
        entries = list(directory.iterdir())
        if len(entries) != 1 or entries[0].is_symlink() or not entries[0].is_file():
            raise CoverageSuccessorError(
                "successor output root contains an unverified or foreign artifact"
            )
        path = entries[0]
        prefix, schema = expected[directory.name]
        document = _addressed(path)
        digest = str(document["artifact_sha256"])
        if path.name != f"{prefix}-{digest}.json" or document.get("schema_version") != schema:
            raise CoverageSuccessorError(
                "successor output root contains an unverified or foreign artifact"
            )
        documents[directory.name] = document

    plan = documents["plan"]
    preflight = documents["preflight"]
    dry_run = documents["dry-run"]
    template = documents["templates"]
    verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    if (
        plan.get("freeze_nonce") != FREEZE_NONCE
        or plan.get("supersedes_failed_offline_plan_sha256") != FAILED_V3_PLAN_SHA256
        or plan.get("failed_offline_supersession") != _failed_v3_freeze_evidence(repo_root)
        or plan.get("supersedes_retired_format_only_v4_plan_sha256")
        != RETIRED_V4_PLAN_SHA256
        or plan.get("retired_format_only_v4_freeze")
        != _retired_v4_format_freeze_evidence(repo_root)
        or preflight.get("plan_sha256") != plan.get("artifact_sha256")
        or preflight.get("decision") != "execution_not_admitted"
        or preflight.get("calls_made")
        != {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
        or dry_run.get("plan_sha256") != plan.get("artifact_sha256")
        or dry_run.get("preflight_sha256") != preflight.get("artifact_sha256")
        or (dry_run.get("counts") or {}).get("provider_completions") != 0
        or (dry_run.get("counts") or {}).get("catalog_gets") != 0
        or (dry_run.get("counts") or {}).get("epicure_calls") != 0
        or template.get("plan_sha256") != plan.get("artifact_sha256")
        or template.get("work_item_ids") != _ordered_cohere_work_item_ids(plan)
    ):
        raise CoverageSuccessorError("existing successor v4 offline artifact set does not verify")
    return documents


def _discover_existing_v4_offline_outputs(
    *,
    repo_root: Path,
    output_root: Path,
    existing_output: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[tuple[Path, ...], dict[str, Any] | None]:
    """Find only complete, fully verified V4 offline freezes.

    Equivalent freezes are ownership aliases, not prior executions.  An incomplete,
    foreign, or non-identical V4 root fails closed; this is what permits a second
    alternate-output freeze without weakening execution-identity collision checks.
    """

    verified: dict[Path, dict[str, dict[str, Any]]] = {}
    if existing_output is not None:
        verified[output_root.resolve()] = {
            key: dict(value) for key, value in existing_output.items()
        }
    for governed_root in (repo_root / "flavourbench/artifacts", repo_root / "artifacts"):
        if not governed_root.exists():
            continue
        for path in sorted(
            governed_root.rglob("frontier-coverage-primary-successor-plan-*.json")
        ):
            document = _regular_json(path)
            if document.get("schema_version") != PLAN_SCHEMA:
                continue
            candidate_root = path.parent.parent.resolve()
            if candidate_root == output_root.resolve() and existing_output is None:
                continue
            if candidate_root in verified:
                continue
            candidate = _verify_existing_offline_output_root(
                output_root=candidate_root, repo_root=repo_root
            )
            if candidate is None:
                raise CoverageSuccessorError("discovered V4 offline root is empty")
            verified[candidate_root] = candidate
    plans = [documents["plan"] for documents in verified.values()]
    if plans and any(plan != plans[0] for plan in plans[1:]):
        raise CoverageSuccessorError("multiple non-identical successor v4 freezes exist")
    return tuple(sorted(verified, key=lambda path: path.as_posix())), (plans[0] if plans else None)


def _prior_identifiers(
    root: Path,
    *,
    verified_successor_output: Path,
    successor_plan: Mapping[str, Any] | None = None,
    additional_verified_successor_outputs: Sequence[Path] = (),
) -> set[str]:
    singular = {
        "batch_id",
        "cell_id",
        "work_item_id",
        "dataset_work_item_id",
        "run_id",
        "arm_id",
        "attempt_id",
    }
    plural = {
        "batch_ids",
        "cell_ids",
        "work_item_ids",
        "run_ids",
        "arm_ids",
        "attempt_ids",
        "retired_predecessor_ids",
    }
    found: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if key in singular and isinstance(value, str) and value:
            found.add(value)
        elif key in plural and isinstance(value, Mapping):
            found.update(str(item) for item in value.values() if isinstance(item, str))
        elif key in plural and isinstance(value, list):
            found.update(str(item) for item in value if isinstance(item, str))
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    documents: dict[Path, object] = {}
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise CoverageSuccessorError(
                f"prior-identifier inventory found non-regular JSON: {path}"
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CoverageSuccessorError(
                f"prior-identifier inventory cannot parse JSON: {path}"
            ) from error
        documents[path] = document

    output = verified_successor_output.resolve()
    additional_outputs = tuple(
        path.resolve()
        for path in additional_verified_successor_outputs
        if path.resolve() != output
    )
    owned_plan = dict(successor_plan) if successor_plan is not None else None
    stored_plans = [
        document
        for path, document in documents.items()
        if path.resolve().is_relative_to(output)
        and isinstance(document, Mapping)
        and document.get("schema_version") == PLAN_SCHEMA
    ]
    if len(stored_plans) > 1:
        raise CoverageSuccessorError("successor output contains multiple plans")
    if owned_plan is None and stored_plans:
        owned_plan = dict(stored_plans[0])
    if owned_plan is not None:
        body = {key: value for key, value in owned_plan.items() if key != "artifact_sha256"}
        if (
            owned_plan.get("artifact_sha256") != _sha256(body)
            or owned_plan.get("schema_version") != PLAN_SCHEMA
            or owned_plan.get("freeze_nonce") != FREEZE_NONCE
            or (owned_plan.get("supersedes") or {}).get("semantic_sha256") != PREDECESSOR_SHA256
            or (
                stored_plans
                and stored_plans[0].get("artifact_sha256") != owned_plan.get("artifact_sha256")
            )
        ):
            raise CoverageSuccessorError("successor-owned plan does not verify")

    owned_plan_sha = str((owned_plan or {}).get("artifact_sha256") or "")
    owned_cells = {
        str(cell.get("work_item_id") or ""): cell
        for cell in ((owned_plan or {}).get("cells") or [])
        if isinstance(cell, Mapping)
    }
    owned_batches = {
        str(batch.get("batch_id") or ""): batch
        for batch in ((owned_plan or {}).get("endpoint_batches") or [])
        if isinstance(batch, Mapping)
    }
    owned_journals: set[Path] = set()
    preflight_documents = [
        document
        for path, document in documents.items()
        if path.resolve().is_relative_to(output)
        and isinstance(document, Mapping)
        and document.get("schema_version") == PREFLIGHT_SCHEMA
    ]
    if len(preflight_documents) > 1:
        raise CoverageSuccessorError("successor output contains multiple preflights")
    owned_preflight_sha = str(
        (preflight_documents[0] if preflight_documents else {}).get("artifact_sha256") or ""
    )
    blocker_codes = {
        str(blocker.get("code") or "")
        for blocker in (
            (preflight_documents[0] if preflight_documents else {}).get("blockers") or []
        )
        if isinstance(blocker, Mapping)
    }
    receipt_bindings: list[tuple[str, str]] = []
    for path, document in documents.items():
        if any(path.resolve().is_relative_to(root) for root in additional_outputs):
            continue
        inside_successor = path.resolve().is_relative_to(output)
        if inside_successor:
            if not isinstance(document, Mapping):
                raise CoverageSuccessorError("successor output contains non-object JSON")
            schema = str(document.get("schema_version") or "")
            body = {key: value for key, value in document.items() if key != "artifact_sha256"}
            valid_hash = document.get("artifact_sha256") == _sha256(body)
            keys = set(map(str, document))
            binding = False
            if schema == PLAN_SCHEMA:
                binding = bool(valid_hash and document.get("artifact_sha256") == owned_plan_sha)
            elif schema == PREFLIGHT_SCHEMA:
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("decision") == "execution_not_admitted"
                    and document.get("calls_made")
                    == {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
                    and keys
                    == {
                        "schema_version",
                        "status",
                        "plan_sha256",
                        "source_closure_sha256",
                        "environment_sha256",
                        "checks",
                        "budget",
                        "support",
                        "blockers",
                        "decision",
                        "calls_made",
                        "claim_boundary",
                        "artifact_sha256",
                    }
                )
            elif schema == DRY_RUN_SCHEMA:
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("preflight_sha256") == owned_preflight_sha
                    and (document.get("counts") or {}).get("provider_completions") == 0
                    and (document.get("counts") or {}).get("epicure_calls") == 0
                    and keys
                    == {
                        "schema_version",
                        "status",
                        "plan_sha256",
                        "preflight_sha256",
                        "executor_entrypoint",
                        "source_closure_sha256",
                        "decisions",
                        "counts",
                        "support",
                        "budget",
                        "blockers",
                        "claim_boundary",
                        "artifact_sha256",
                    }
                )
            elif schema == "flavourbench-frontier-coverage-primary-successor-receipt-v1":
                batch_id = str(document.get("batch_id") or "")
                outcomes = document.get("outcomes") or []
                head = str(
                    (document.get("ledger_terminal_prefix") or {}).get("head_entry_sha256") or ""
                )
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and batch_id in owned_batches
                    and isinstance(outcomes, list)
                    and {str(row.get("work_item_id") or "") for row in outcomes}
                    == set(owned_batches[batch_id]["work_item_ids"])
                    and head
                    and keys
                    == {
                        "schema_version",
                        "status",
                        "plan_sha256",
                        "preflight_sha256",
                        "live_admission_sha256",
                        "batch_id",
                        "ledger_terminal_prefix",
                        "outcomes",
                        "observed_model_pair_family_support",
                        "provider_or_epicure_calls_made_by_receipt",
                        "claim_boundary",
                        "artifact_sha256",
                    }
                )
                if binding:
                    receipt_bindings.append((batch_id, head))
            elif schema == "flavourbench-frontier-coverage-primary-live-admission-v1":
                batch_id = str(document.get("authorized_batch_id") or "")
                reservations = document.get("batch_reservations") or {}
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("preflight_sha256") == owned_preflight_sha
                    and batch_id in owned_batches
                    and isinstance(reservations, Mapping)
                    and set(map(str, reservations)) == {batch_id}
                    and (reservations[batch_id] or {}).get("work_item_ids")
                    == owned_batches[batch_id]["work_item_ids"]
                    and keys
                    == {
                        "schema_version",
                        "plan_sha256",
                        "preflight_sha256",
                        "authorized_batch_id",
                        "development_only",
                        "official",
                        "rank_eligible",
                        "blocker_closures",
                        "batch_reservations",
                        "provider_or_epicure_calls_made_by_admission",
                        "artifact_sha256",
                    }
                )
            elif schema == "flavourbench-frontier-coverage-primary-blocker-evidence-v1":
                code = str(document.get("blocker_code") or "")
                specific = {
                    "live_route_availability_not_tested": "route_records",
                    "reasoning_effort_sensitivity_precedes_coverage": "ordering_released",
                    "cross_study_budget_contention_requires_locked_rebase": (
                        "locked_rebase_required_at_execution"
                    ),
                    "epicure_lineage_not_independently_reconstructable": "epicure",
                    "independent_governance_go_required": "independent_governance_decision",
                    "successor_execution_root_not_empty": "run_roots_reset",
                }.get(code)
                binding = bool(
                    valid_hash
                    and code in blocker_codes
                    and specific
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("preflight_sha256") == owned_preflight_sha
                    and keys
                    <= {
                        "schema_version",
                        "blocker_code",
                        "plan_sha256",
                        "preflight_sha256",
                        "decision",
                        "route_records",
                        "ordering_released",
                        "locked_rebase_required_at_execution",
                        "epicure",
                        "execution_scope",
                        "official_release_blocker_remains",
                        "independent_governance_decision",
                        "run_roots_reset",
                        "artifact_sha256",
                    }
                    and specific in keys
                )
            elif schema == "flavourbench-cohere-direct-resource-envelope-v1":
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("preflight_sha256") == owned_preflight_sha
                    and document.get("resource_envelope")
                    == (owned_plan or {}).get("cohere_prospective_resource_envelope")
                    and keys
                    == {
                        "schema_version",
                        "blocker_code",
                        "plan_sha256",
                        "preflight_sha256",
                        "decision",
                        "resource_envelope",
                        "operator_attestation",
                        "artifact_sha256",
                    }
                )
            elif schema == COHERE_OPERATOR_ATTESTATION_SCHEMA:
                public_binding = document.get("credential_binding_public_object")
                operator = document.get("operator")
                binding = bool(
                    valid_hash
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("status") == "operator_authorized_exact_resource_scope"
                    and document.get("decision") == "authorize_exact_bounded_cohere_scholars_use"
                    and document.get("provider") == "cohere_direct"
                    and document.get("credential_program") == "Cohere Scholars"
                    and document.get("work_item_ids")
                    == _ordered_cohere_work_item_ids(owned_plan or {})
                    and document.get("resource_envelope_sha256")
                    == (owned_plan or {})
                    .get("cohere_prospective_resource_envelope", {})
                    .get("envelope_sha256")
                    and document.get("credential_binding_method")
                    == "sha256_canonical_public_binding_object"
                    and isinstance(public_binding, Mapping)
                    and document.get("credential_binding_sha256") == _sha256(public_binding)
                    and document.get("credential_binding_is_derived_from_secret") is False
                    and document.get("contains_secret") is False
                    and document.get("usd_cost_or_reservation_claimed") is False
                    and document.get("provider_or_epicure_calls_made_by_attestation") is False
                    and isinstance(operator, Mapping)
                    and set(operator) == {"full_name", "role"}
                    and not _contains_secret_material(document)
                    and keys
                    == {
                        "schema_version",
                        "status",
                        "plan_sha256",
                        "decision",
                        "credential_program",
                        "provider",
                        "operator",
                        "issued_at",
                        "expires_at",
                        "work_item_ids",
                        "resource_envelope_sha256",
                        "credential_binding_method",
                        "credential_binding_public_object",
                        "credential_binding_sha256",
                        "credential_binding_is_derived_from_secret",
                        "contains_secret",
                        "usd_cost_or_reservation_claimed",
                        "provider_or_epicure_calls_made_by_attestation",
                        "artifact_sha256",
                    }
                )
            elif schema == COHERE_OPERATOR_TEMPLATE_SCHEMA:
                binding = bool(
                    valid_hash
                    and document.get("status") == "template_not_authorization"
                    and document.get("plan_sha256") == owned_plan_sha
                    and document.get("provider") == "cohere_direct"
                    and document.get("credential_program") == "Cohere Scholars"
                    and document.get("work_item_ids")
                    == _ordered_cohere_work_item_ids(owned_plan or {})
                    and document.get("resource_envelope_sha256")
                    == (owned_plan or {})
                    .get("cohere_prospective_resource_envelope", {})
                    .get("envelope_sha256")
                    and document.get("execution_policy_sha256")
                    == (owned_plan or {}).get("execution_policy", {}).get("execution_policy_sha256")
                    and document.get("contains_secret") is False
                    and document.get("usd_cost_or_reservation_claimed") is False
                    and not _contains_secret_material(document)
                    and keys
                    == {
                        "schema_version",
                        "status",
                        "plan_sha256",
                        "provider",
                        "credential_program",
                        "work_item_ids",
                        "resource_envelope_sha256",
                        "resource_envelope_totals",
                        "execution_policy_sha256",
                        "required_attestation_fields",
                        "credential_binding_rule",
                        "prohibitions",
                        "contains_secret",
                        "usd_cost_or_reservation_claimed",
                        "artifact_sha256",
                    }
                )
            if not binding:
                raise CoverageSuccessorError(
                    "successor output root contains an unverified or foreign artifact"
                )
            continue
        if (
            owned_plan is not None
            and isinstance(document, Mapping)
            and document.get("schema_version") == "flavourbench-live-smoke-v1"
            and str(document.get("dataset_work_item_id") or "") in owned_cells
        ):
            work_id = str(document["dataset_work_item_id"])
            cell = owned_cells[work_id]
            from .frontier_contract_runner import _verify_live_artifact

            try:
                verified_source, verified_digest = _verify_live_artifact(path)
            except Exception as error:
                raise CoverageSuccessorError(
                    "successor-owned live source content address does not verify"
                ) from error
            if verified_digest != document.get("artifact_sha256"):
                raise CoverageSuccessorError("successor-owned live source digest differs")
            document = verified_source
            attempts = document.get("provider_attempt_events") or []
            allowed_attempts = {str(slot["attempt_id"]) for slot in cell["attempt_slots"]}
            if (
                document.get("run_id") != cell["run_id"]
                or document.get("dataset_task_id") != cell["task_id"]
                or document.get("requested_conditions") != ["epicure_on"]
                or document.get("candidate_manifest_sha256") != cell["route_manifest_sha256"]
                or not isinstance(attempts, list)
                or any(
                    not isinstance(event, Mapping)
                    or str(event.get("attempt_id") or "") not in allowed_attempts
                    or str(event.get("arm_id") or "") != cell["arm_ids"]["epicure_on"]
                    for event in attempts
                )
            ):
                raise CoverageSuccessorError("successor-owned live source does not verify")
            descriptor = document.get("run_journal")
            if not isinstance(descriptor, Mapping):
                raise CoverageSuccessorError(
                    "successor-owned source lacks a required run-journal descriptor"
                )
            from .run_journal import verify_journal_descriptor

            try:
                journal_entries = verify_journal_descriptor(path.parent, descriptor)
            except Exception as error:
                raise CoverageSuccessorError(
                    "successor-owned source journal does not verify"
                ) from error
            journal_attempts = [
                entry.get("payload")
                for entry in journal_entries
                if entry.get("event_type") == "provider_attempt"
            ]
            journal_tools = [
                entry.get("payload")
                for entry in journal_entries
                if entry.get("event_type") == "mcp_trace"
            ]
            started = journal_entries[0].get("payload") or {}
            if (
                descriptor.get("run_id") != cell["run_id"]
                or not isinstance(started, Mapping)
                or started.get("dataset_work_item_id") != work_id
                or started.get("dataset_task_id") != cell["task_id"]
                or started.get("candidate_manifest_sha256") != cell["route_manifest_sha256"]
                or started.get("prompt_sha256") != cell["prompt_sha256"]
                or started.get("epicure_conditions") != ["epicure_on"]
                or journal_attempts != document.get("provider_attempt_events")
                or journal_tools != document.get("mcp_trace_events")
            ):
                raise CoverageSuccessorError(
                    "successor-owned source journal binding differs from the artifact"
                )
            owned_journals.add((path.parent / str(descriptor["filename"])).resolve())
            continue
        visit(document)
    coordinator_terminal_heads: dict[str, str] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise CoverageSuccessorError(
                f"prior-identifier inventory found non-regular JSONL: {path}"
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise CoverageSuccessorError(
                f"prior-identifier inventory cannot read JSONL: {path}"
            ) from error
        resolved = path.resolve()
        if any(resolved.is_relative_to(root) for root in additional_outputs):
            continue
        if resolved in owned_journals:
            continue
        inside_successor = resolved.is_relative_to(output)
        parsed_lines: list[dict[str, Any]] = []
        previous: str | None = None
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                raise CoverageSuccessorError(
                    f"prior-identifier inventory found a blank JSONL line: {path}:{number}"
                )
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise CoverageSuccessorError(
                    f"prior-identifier inventory cannot parse {path}:{number}"
                ) from error
            if not isinstance(entry, dict):
                raise CoverageSuccessorError(
                    f"prior-identifier inventory found a non-object: {path}:{number}"
                )
            parsed_lines.append(entry)
            if inside_successor or (
                owned_plan_sha
                and entry.get("schema_version") == "flavourbench-live-run-journal-v1"
                and str(entry.get("run_id") or "")
                in {str(cell["run_id"]) for cell in owned_cells.values()}
            ):
                if (
                    entry.get("sequence") != number
                    or entry.get("previous_entry_sha256") != previous
                    or entry.get("entry_sha256")
                    != _sha256(
                        {key: value for key, value in entry.items() if key != "entry_sha256"}
                    )
                ):
                    raise CoverageSuccessorError(
                        f"successor-owned JSONL hash chain fails: {path}:{number}"
                    )
                previous = str(entry["entry_sha256"])
        if inside_successor:
            expected_path = (output / "run/coordinator/ledger.jsonl").resolve()
            if resolved != expected_path or not owned_plan_sha:
                raise CoverageSuccessorError("successor output contains a foreign JSONL")
            for entry in parsed_lines:
                event_type = str(entry.get("event_type") or "")
                batch_id = str(entry.get("batch_id") or "")
                work_id = str(entry.get("work_item_id") or "")
                cell = owned_cells.get(work_id) if work_id else None
                if (
                    entry.get("schema_version")
                    != "flavourbench-frontier-coverage-primary-successor-ledger-v1"
                    or entry.get("plan_sha256") != owned_plan_sha
                    or event_type
                    not in {
                        "endpoint_batch_reserved",
                        "item_execution_started",
                        "item_terminalized",
                        "execution_incident",
                        "endpoint_batch_terminalized",
                    }
                    or batch_id not in owned_batches
                    or (work_id and cell is None)
                    or (
                        cell is not None and work_id not in owned_batches[batch_id]["work_item_ids"]
                    )
                    or (
                        entry.get("run_id") is not None
                        and (cell is None or entry.get("run_id") != cell["run_id"])
                    )
                    or (
                        entry.get("arm_id") is not None
                        and (cell is None or entry.get("arm_id") != cell["arm_ids"]["epicure_on"])
                    )
                    or (
                        entry.get("attempt_id") is not None
                        and (
                            cell is None
                            or entry.get("attempt_id")
                            not in {slot["attempt_id"] for slot in cell["attempt_slots"]}
                        )
                    )
                    or (
                        event_type == "item_execution_started"
                        and (
                            cell is None
                            or entry.get("attempt_slots_sha256") != cell["attempt_slots_sha256"]
                        )
                    )
                ):
                    raise CoverageSuccessorError("successor coordinator ledger binding differs")
                if entry.get("event_type") == "endpoint_batch_terminalized":
                    if batch_id in coordinator_terminal_heads:
                        raise CoverageSuccessorError("successor coordinator terminal is duplicated")
                    coordinator_terminal_heads[batch_id] = str(entry["entry_sha256"])
            continue
        if any(
            entry.get("schema_version") == "flavourbench-frontier-contract-ledger-v1"
            and (
                entry.get("coverage_successor_plan_sha256") == owned_plan_sha
                or entry.get("runner_run_id") == owned_plan_sha
            )
            for entry in parsed_lines
        ):
            previous = None
            for number, entry in enumerate(parsed_lines, start=1):
                if (
                    entry.get("sequence") != number
                    or entry.get("previous_entry_sha256") != previous
                    or entry.get("entry_sha256")
                    != _sha256(
                        {key: value for key, value in entry.items() if key != "entry_sha256"}
                    )
                ):
                    raise CoverageSuccessorError(
                        f"canonical frontier ledger hash chain fails: {path}:{number}"
                    )
                previous = str(entry["entry_sha256"])
        if parsed_lines and all(
            entry.get("schema_version") == "flavourbench-live-run-journal-v1"
            and str(entry.get("run_id") or "")
            in {str(cell["run_id"]) for cell in owned_cells.values()}
            for entry in parsed_lines
        ):
            run_id = str(parsed_lines[0]["run_id"])
            candidates = [cell for cell in owned_cells.values() if cell["run_id"] == run_id]
            if len(candidates) != 1:
                raise CoverageSuccessorError("successor live journal run identity is ambiguous")
            cell = candidates[0]
            allowed_attempts = {str(slot["attempt_id"]) for slot in cell["attempt_slots"]}
            start_payload = parsed_lines[0].get("payload") or {}
            if (
                parsed_lines[0].get("event_type") != "run_started"
                or not isinstance(start_payload, Mapping)
                or start_payload.get("dataset_work_item_id") != cell["work_item_id"]
                or start_payload.get("dataset_task_id") != cell["task_id"]
                or start_payload.get("candidate_manifest_sha256") != cell["route_manifest_sha256"]
                or start_payload.get("epicure_conditions") != ["epicure_on"]
            ):
                raise CoverageSuccessorError("successor live journal start binding differs")
            for entry in parsed_lines[1:]:
                payload = entry.get("payload") or {}
                if not isinstance(payload, Mapping):
                    raise CoverageSuccessorError("successor live journal payload is malformed")
                if entry.get("event_type") == "provider_attempt" and (
                    str(payload.get("attempt_id") or "") not in allowed_attempts
                    or payload.get("arm_id") != cell["arm_ids"]["epicure_on"]
                ):
                    raise CoverageSuccessorError(
                        "successor live journal provider-attempt binding differs"
                    )
                if entry.get("event_type") == "mcp_trace" and (
                    payload.get("arm_id") != cell["arm_ids"]["epicure_on"]
                ):
                    raise CoverageSuccessorError("successor live journal MCP binding differs")
            continue
        for entry in parsed_lines:
            if (
                owned_plan_sha
                and entry.get("schema_version") == "flavourbench-frontier-contract-ledger-v1"
                and (
                    entry.get("coverage_successor_plan_sha256") == owned_plan_sha
                    or entry.get("runner_run_id") == owned_plan_sha
                )
            ):
                work_id = str(entry.get("coverage_successor_work_item_id") or "")
                batch_id = str(entry.get("coverage_successor_batch_id") or "")
                cell = owned_cells.get(work_id)
                if (
                    cell is None
                    or batch_id not in owned_batches
                    or work_id not in owned_batches[batch_id]["work_item_ids"]
                    or entry.get("model_id") != cell["model_id"]
                    or entry.get("provider_tag") != cell["provider_tag"]
                    or entry.get("manifest_sha256") != cell["route_manifest_sha256"]
                ):
                    raise CoverageSuccessorError("canonical successor ledger binding differs")
                continue
            visit(entry)
    if any(coordinator_terminal_heads.get(batch_id) != head for batch_id, head in receipt_bindings):
        raise CoverageSuccessorError("successor receipt does not bind a terminal ledger entry")
    found.discard("")
    return found


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(body)
    document = {**body, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise CoverageSuccessorError(f"content-addressed artifact conflict: {destination}")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as out:
        temporary = Path(out.name)
        out.write(rendered)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def _candidate_routes(repo_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    routes: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for relative in ROUTE_MANIFESTS:
        path = repo_root / relative
        manifest = load_candidate_manifest(path, expected_digest="")
        digest = str(manifest["content_address"]["digest"])
        reference = _file_ref(repo_root, path, expected_sha256=digest)
        references.append(reference)
        raw_models = manifest.get("models")
        if not isinstance(raw_models, list):
            raise CoverageSuccessorError(f"route manifest has no model records: {path}")
        raw_by_id: dict[str, Mapping[str, Any]] = {}
        for index, raw in enumerate(raw_models):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
                raise CoverageSuccessorError(
                    f"route manifest model record {index} is malformed: {path}"
                )
            raw_model_id = str(raw["model"].get("id") or "")
            if not raw_model_id or raw_model_id in raw_by_id:
                raise CoverageSuccessorError(
                    f"route manifest has an absent or duplicate model ID: {path}"
                )
            raw_by_id[raw_model_id] = raw
        for candidate in select_candidates(manifest):
            if candidate.model_id in routes:
                raise CoverageSuccessorError(f"duplicated exact route: {candidate.model_id}")
            raw = raw_by_id.get(candidate.model_id)
            if raw is None:
                raise CoverageSuccessorError(
                    f"selected candidate lacks its raw route record: {candidate.model_id}"
                )
            request_policy = raw.get("request_policy")
            request_provider = (
                request_policy.get("provider") if isinstance(request_policy, Mapping) else None
            )
            if not isinstance(request_provider, Mapping):
                raise CoverageSuccessorError(
                    f"route lacks its exact provider policy: {candidate.model_id}"
                )
            only = request_provider.get("only") if isinstance(request_provider, Mapping) else None
            if (
                candidate.execution_backend not in {"openrouter", "kimi_direct", "cohere_direct"}
                or candidate.route_selection.get("selection_frozen_before_generation") is not True
                or candidate.route_selection.get("generation_time_automatic_fallback") is not False
                or not isinstance(only, list)
                or only != [candidate.provider_tag]
                or request_provider.get("allow_fallbacks") is not False
                or request_provider.get("require_parameters") is not True
                or request_provider.get("data_collection") != "deny"
                or candidate.endpoint.get("model_id") != candidate.model_id
                or not candidate.canonical_model_slug
                or not candidate.backend_contract_sha256
            ):
                raise CoverageSuccessorError(
                    f"route is not exact and fallback-free: {candidate.model_id}"
                )
            pricing = candidate.endpoint.get("pricing") or {}
            status = str(pricing.get("status") or "")
            zero_prices = all(
                _decimal(pricing.get(field, "0"), field=f"{candidate.model_id} {field}") == 0
                for field in ("prompt", "completion", "request")
            )
            is_non_usd_cohere = candidate.execution_backend == "cohere_direct" and zero_prices
            cost_status = NON_USD_UNKNOWN_STATUS if is_non_usd_cohere else "priced_frozen_route"
            historical_pricing = {
                "not_current_pricing_budget_or_free_tier_claim": True,
                "source_manifest_status_label": status,
                "source_manifest_numeric_fields": {
                    field: str(pricing.get(field) or "0")
                    for field in ("prompt", "completion", "internal_reasoning", "request")
                },
                "interpretation": (
                    "source_only_unpriced_provenance_not_a_current_zero_price_or_free_tier_claim"
                ),
            }
            routes[candidate.model_id] = {
                "manifest": reference,
                "canonical_model_slug": candidate.canonical_model_slug,
                "execution_backend": candidate.execution_backend,
                "provider_tag": candidate.provider_tag,
                "backend_contract_sha256": candidate.backend_contract_sha256,
                "endpoint_document_sha256": candidate.endpoint_sha256,
                "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
                "expected_actual_provider": candidate.endpoint.get("provider_name"),
                "request_provider_only": list(only),
                "requested_model_id": candidate.model_id,
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "pricing_status": NON_USD_UNKNOWN_STATUS if is_non_usd_cohere else status,
                "pricing_currency": None if is_non_usd_cohere else "USD",
                "pricing": {
                    field: None if is_non_usd_cohere else str(pricing.get(field) or "0")
                    for field in ("prompt", "completion", "internal_reasoning", "request")
                },
                **(
                    {"historical_source_provenance": historical_pricing}
                    if is_non_usd_cohere
                    else {}
                ),
                "cost_status": cost_status,
                "offline_route_identity_verified": True,
                "live_route_availability_verified": False,
            }
    return routes, sorted(references, key=lambda value: value["semantic_sha256"])


def _validated_baseline(repo_root: Path) -> tuple[Decimal, dict[str, Any]]:
    preflight = _addressed(
        repo_root / PREDECESSOR_PREFLIGHT, expected_sha256=PREDECESSOR_PREFLIGHT_SHA256
    )
    gate = _addressed(repo_root / COHERE_GATE, expected_sha256=COHERE_GATE_SHA256)
    calls = preflight.get("calls") or {}
    decision = gate.get("decision") or {}
    gate_calls = gate.get("calls") or {}
    budget = preflight.get("budget") or {}
    baseline = _decimal(budget.get("rebased_current_exposure_usd"), field="bound baseline exposure")
    if (
        preflight.get("schema_version") != "flavourbench-frontier-coverage-primary-preflight-v1"
        or preflight.get("status") != "budget_fits_but_blocked_pending_independent_governance_go"
        or (preflight.get("primary_plan") or {}).get("artifact_sha256") != PREDECESSOR_SHA256
        or calls != {"epicure": 0, "provider": 0}
        or budget.get("admission_granted") is not False
        or baseline != CURRENT_EXPOSURE_USD
        or gate.get("schema_version") != "flavourbench-cohere-continuation-route-gate-v1"
        or gate.get("status") != "passed_offline_transport_contract_only"
        or gate_calls != {"epicure": 0, "provider": 0}
        or decision.get("provider_connectivity_tested") is not False
        or decision.get("provider_semantics_claimed") is not False
        or decision.get("paid_execution_admission_granted") is not False
    ):
        raise CoverageSuccessorError("bound predecessor accounting or zero-call chain differs")
    return baseline, {
        "derivation": "bound_predecessor_preflight_accounting_chain",
        "preflight_sha256": PREDECESSOR_PREFLIGHT_SHA256,
        "offline_cohere_gate_sha256": COHERE_GATE_SHA256,
        "live_source_snapshot": preflight.get("live_source_snapshot"),
        "provider_calls": 0,
        "catalog_calls": 0,
        "epicure_calls": 0,
    }


def build_plan(*, repo_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    output_root = output_root or (repo_root / DEFAULT_OUTPUT_ROOT)
    existing_output = _verify_existing_offline_output_root(
        output_root=output_root, repo_root=repo_root
    )
    owned_output_roots, owned_existing_plan = _discover_existing_v4_offline_outputs(
        repo_root=repo_root,
        output_root=output_root,
        existing_output=existing_output,
    )
    predecessor_path = repo_root / PREDECESSOR
    predecessor = _addressed(predecessor_path, expected_sha256=PREDECESSOR_SHA256)
    if (
        predecessor.get("schema_version") != "flavourbench-frontier-coverage-primary-on-v5-plan-v2"
        or predecessor.get("counts", {}).get("primary_fresh_cells") != 50
    ):
        raise CoverageSuccessorError("predecessor is not the exact 50-cell plan")
    arena_path = repo_root / CORRECTED_ARENA
    arena = _addressed(arena_path, expected_sha256=CORRECTED_ARENA_SHA256)
    task_path = repo_root / TASK_VALIDITY
    tasks, task_source = load_development_task_inventory(task_path)
    if task_source.get("artifact_sha256") != TASK_VALIDITY_SHA256:
        raise CoverageSuccessorError("task-validity loader returned a different dossier")
    task_map = {task.public_id: task for task in tasks}
    routes, manifest_refs = _candidate_routes(repo_root)
    if set(routes) != {str(cell["model_id"]) for cell in predecessor["cells"]}:
        raise CoverageSuccessorError("exact route set differs from predecessor model panel")

    baseline, baseline_derivation = _validated_baseline(repo_root)
    quarantine_path = repo_root / TASK_QUARANTINE
    quarantine = _addressed(quarantine_path, expected_sha256=TASK_QUARANTINE_SHA256)
    quarantined = {
        str(value)
        for key in ("quarantined_task_ids", "task_ids")
        for value in (quarantine.get(key) or [])
    }
    for row in (
        quarantine.get("records")
        or quarantine.get("tasks")
        or quarantine.get("quarantined_tasks")
        or []
    ):
        if isinstance(row, Mapping):
            quarantined.add(str(row.get("task_id") or ""))
    anchor_ids = {str(cell["task_id"]) for cell in predecessor["cells"]}
    if len(anchor_ids) != 4 or anchor_ids & quarantined:
        raise CoverageSuccessorError("coverage anchor set intersects current task quarantine")
    governed_roots = (repo_root / "flavourbench/artifacts", repo_root / "artifacts")
    prior_identifiers: set[str] = set()
    for governed_root in governed_roots:
        if governed_root.exists():
            prior_identifiers.update(
                _prior_identifiers(
                    governed_root,
                    verified_successor_output=output_root,
                    successor_plan=owned_existing_plan,
                    additional_verified_successor_outputs=owned_output_roots,
                )
            )
    retired_identifiers: set[str] = set()
    cells: list[dict[str, Any]] = []
    work_ids: set[str] = set()
    run_ids: set[str] = set()
    arm_ids: set[str] = set()
    attempt_ids: set[str] = set()
    priced_total = Decimal(0)
    for source in predecessor["cells"]:
        cell = dict(source)
        predecessor_cell_id = str(cell["cell_id"])
        predecessor_work_id = str(cell["work_item_id"])
        predecessor_run_id = str(cell["run_id"])
        predecessor_declared_arm = str((cell.get("arm_ids") or {}).get("epicure_on") or "")
        predecessor_slots = list(cell.get("attempt_slots") or [])
        predecessor_slot_arms = {str(slot.get("arm_id") or "") for slot in predecessor_slots}
        predecessor_attempts = [str(slot.get("attempt_id") or "") for slot in predecessor_slots]
        if (
            len(predecessor_declared_arm) != 64
            or predecessor_slot_arms != {f"{predecessor_run_id}:epicure_on"}
            or predecessor_declared_arm in predecessor_slot_arms
            or len(predecessor_slots) != 29
            or len(predecessor_attempts) != len(set(predecessor_attempts))
            or any(not value for value in predecessor_attempts)
        ):
            raise CoverageSuccessorError("predecessor arm mismatch does not have one safe repair")
        task = task_map.get(str(cell["task_id"]))
        if (
            task is None
            or task.family != cell["task_family"]
            or task.prompt_sha256 != cell["prompt_sha256"]
            or task_source.get("rank_eligible") is not False
            or task_source.get("confirmatory_eligible") is not False
        ):
            raise CoverageSuccessorError("coverage anchor task binding changed")
        route = routes[str(cell["model_id"])]
        if (
            route["manifest"]["semantic_sha256"] != cell["route_manifest_sha256"]
            or route["provider_tag"] != cell["provider_tag"]
            or route["execution_backend"] != cell["execution_backend"]
            or route["endpoint_execution_sha256"] != cell["endpoint_execution_sha256"]
        ):
            raise CoverageSuccessorError("cell route differs from its exact manifest")
        cost_status = route["cost_status"]
        predecessor_forecast = _decimal(
            cell["reserved_worst_case_usd"], field="predecessor cell forecast"
        )
        is_cohere = cell["execution_backend"] == "cohere_direct"
        non_usd_or_unpriced = is_cohere or cost_status.startswith(("non_usd_", "unpriced_"))
        if cell["execution_backend"] == "cohere_direct" and (
            cost_status != NON_USD_UNKNOWN_STATUS or predecessor_forecast != 0
        ):
            raise CoverageSuccessorError("Cohere route lost its non-USD zero-forecast provenance")
        if cell["execution_backend"] != "cohere_direct" and non_usd_or_unpriced:
            raise CoverageSuccessorError("non-Cohere route unexpectedly lacks a USD price bound")
        reservation = None if non_usd_or_unpriced else _decimal_text(predecessor_forecast)
        if reservation is not None:
            priced_total += predecessor_forecast
        identity_basis = {
            "schema_version": PLAN_SCHEMA,
            "freeze_nonce": FREEZE_NONCE,
            "predecessor_plan_sha256": PREDECESSOR_SHA256,
            "predecessor_cell_id": predecessor_cell_id,
            "ordinal": cell["ordinal"],
            "model_id": cell["model_id"],
            "task_id": cell["task_id"],
            "task_family": cell["task_family"],
            "route_manifest_sha256": cell["route_manifest_sha256"],
            "endpoint_execution_sha256": cell["endpoint_execution_sha256"],
        }
        cell_id = sha256_json({**identity_basis, "identifier_role": "cell"})
        work_id = sha256_json({**identity_basis, "cell_id": cell_id, "identifier_role": "work"})
        run_id = str(uuid.uuid5(NAMESPACE, f"{FREEZE_NONCE}:run:{work_id}"))
        arm_id = f"{run_id}:epicure_on"
        slots = [
            {
                "arm_id": arm_id,
                "phase": str(slot["phase"]),
                "attempt_index": int(slot["attempt_index"]),
                "attempt_id": str(
                    uuid.uuid5(
                        NAMESPACE,
                        (
                            f"{FREEZE_NONCE}:attempt:{work_id}:{arm_id}:"
                            f"{slot['phase']}:{slot['attempt_index']}"
                        ),
                    )
                ),
            }
            for slot in predecessor_slots
        ]
        slot_attempts = [str(slot["attempt_id"]) for slot in slots]
        item = {
            **{
                key: value
                for key, value in cell.items()
                if key
                not in (
                    {
                        "arm_ids",
                        "attempt_slots",
                        "attempt_slots_sha256",
                        "cell_id",
                        "work_item_id",
                        "run_id",
                    }
                    | ({"reserved_worst_case_usd"} if is_cohere else set())
                )
            },
            "cell_id": cell_id,
            "work_item_id": work_id,
            "run_id": run_id,
            "task": {
                "task_id": task.public_id,
                "family": task.family,
                "prompt": task.prompt,
                "prompt_sha256": task.prompt_sha256,
                "rank_eligible": False,
                "confirmatory_eligible": False,
                "synthetic": False,
            },
            "route": route,
            "arm_ids": {"epicure_on": arm_id},
            "attempt_slots": slots,
            "attempt_slots_sha256": sha256_json(slots),
            "predecessor_lineage": {
                "cell_id": predecessor_cell_id,
                "work_item_id": predecessor_work_id,
                "run_id": predecessor_run_id,
                "declared_arm_id": predecessor_declared_arm,
                "attempt_slot_arm_id": next(iter(predecessor_slot_arms)),
                "attempt_ids_sha256": sha256_json(sorted(predecessor_attempts)),
                "all_predecessor_identities_retired": True,
            },
            "arm_identity_repair": (
                "fresh_run_condition_identity_bound_uniformly_to_all_fresh_attempt_slots"
            ),
            "cost_reservation": (
                {
                    "status": NON_USD_UNKNOWN_STATUS,
                    "currency": None,
                    "successor_reservation_usd": None,
                    "current_usd_price_or_reservation_available": False,
                    "historical_source_provenance": {
                        "not_current_pricing_budget_or_free_tier_claim": True,
                        "predecessor_reserved_worst_case_usd": _decimal_text(
                            predecessor_forecast
                        ),
                        "interpretation": (
                            "source_only_unpriced_provenance_not_a_current_zero_price_or_"
                            "free_tier_claim"
                        ),
                    },
                }
                if is_cohere
                else {
                    "status": cost_status,
                    "predecessor_forecast_usd": _decimal_text(predecessor_forecast),
                    "successor_reservation_usd": reservation,
                    "current_price_claimed": True,
                }
            ),
        }
        if (
            work_id in work_ids
            or run_id in run_ids
            or arm_id in arm_ids
            or attempt_ids.intersection(slot_attempts)
            or prior_identifiers.intersection({cell_id, work_id, run_id, arm_id, *slot_attempts})
        ):
            raise CoverageSuccessorError(
                "successor execution identity is duplicated or overlaps prior records"
            )
        work_ids.add(work_id)
        run_ids.add(run_id)
        arm_ids.add(arm_id)
        attempt_ids.update(slot_attempts)
        retired_identifiers.update(
            {
                predecessor_cell_id,
                predecessor_work_id,
                predecessor_run_id,
                predecessor_declared_arm,
                *predecessor_slot_arms,
                *predecessor_attempts,
            }
        )
        cells.append(item)

    predecessor_to_successor = {
        str(cell["predecessor_lineage"]["work_item_id"]): str(cell["work_item_id"])
        for cell in cells
    }
    batch_source = [
        dict(batch) for batch in predecessor["endpoint_isolated_batches_in_balanced_order"]
    ]
    batches: list[dict[str, Any]] = []
    for batch in sorted(batch_source, key=lambda row: int(row["execution_ordinal"])):
        successor_work_ids = [predecessor_to_successor[value] for value in batch["work_item_ids"]]
        batch_cells = [cell for cell in cells if cell["work_item_id"] in successor_work_ids]
        if len(batch_cells) != int(batch["cell_count"]):
            raise CoverageSuccessorError("endpoint batch membership changed")
        unpriced = [
            cell
            for cell in batch_cells
            if cell["cost_reservation"]["successor_reservation_usd"] is None
        ]
        priced = sum(
            (
                Decimal(cell["cost_reservation"]["successor_reservation_usd"])
                for cell in batch_cells
                if cell["cost_reservation"]["successor_reservation_usd"] is not None
            ),
            Decimal(0),
        )
        batch_id = sha256_json(
            {
                "schema_version": PLAN_SCHEMA,
                "freeze_nonce": FREEZE_NONCE,
                "identifier_role": "endpoint_batch",
                "predecessor_batch_id": batch["batch_id"],
                "execution_ordinal": batch["execution_ordinal"],
                "model_id": batch["model_id"],
                "work_item_ids": successor_work_ids,
            }
        )
        if batch_id in prior_identifiers or batch_id in {value["batch_id"] for value in batches}:
            raise CoverageSuccessorError("successor batch identity overlaps a prior record")
        retired_identifiers.add(str(batch["batch_id"]))
        batch_entry = {
                **{
                    key: value
                    for key, value in batch.items()
                    if key not in {"batch_id", "work_item_ids", "worst_case_usd"}
                },
                "batch_id": batch_id,
                "work_item_ids": successor_work_ids,
                "predecessor_batch_id": batch["batch_id"],
                "successor_priced_reserve_usd": (None if unpriced else _decimal_text(priced)),
                "unpriced_cell_count": len(unpriced),
                "complete_reservation_bound": not unpriced,
                "admission_authorized": False,
            }
        if unpriced:
            if len(unpriced) != len(batch_cells):
                raise CoverageSuccessorError("mixed priced and non-USD endpoint batch")
            batch_entry["historical_source_provenance"] = {
                "not_current_pricing_budget_or_free_tier_claim": True,
                "predecessor_worst_case_usd": str(batch["worst_case_usd"]),
                "interpretation": (
                    "source_only_unpriced_provenance_not_a_current_zero_price_or_free_tier_claim"
                ),
            }
        else:
            batch_entry["predecessor_worst_case_usd"] = batch["worst_case_usd"]
        batches.append(batch_entry)

    source_closure = build_source_closure(repo_root=repo_root)
    before = predecessor["support_reconstruction"]["before"]
    projected = predecessor["support_reconstruction"]["projected_after_all_usable"]
    before_family = before["missing_cells_by_family"]
    if (
        before["comparisons"] != 915
        or before["missing_cells"] != 73
        or before_family
        != {"composition": 3, "cookability": 20, "evidence": 27, "substitution": 23}
        or projected["comparisons"] != 1281
        or projected["missing_cells"] != 0
        or Counter(cell["task_family"] for cell in cells)
        != Counter({"composition": 4, "cookability": 16, "evidence": 15, "substitution": 15})
        or Counter(cell["execution_backend"] for cell in cells)
        != Counter({"openrouter": 39, "kimi_direct": 3, "cohere_direct": 8})
    ):
        raise CoverageSuccessorError("support or route projection differs from predecessor")
    epicure = arena.get("epicure")
    if not isinstance(epicure, Mapping):
        raise CoverageSuccessorError("corrected arena has no Epicure provenance")
    cohere_cells = [cell for cell in cells if cell["execution_backend"] == "cohere_direct"]
    cohere_cell_limits: dict[str, dict[str, Any]] = {}
    for cell in cohere_cells:
        provider_phases = {
            "planning",
            "tool_round_0",
            "tool_round_1",
            "tool_round_2",
            "final",
        }
        provider_attempt_slots = sum(
            str(slot["phase"]) in provider_phases for slot in cell["attempt_slots"]
        )
        mcp_session_slots = sum(
            str(slot["phase"]) == "mcp_session" for slot in cell["attempt_slots"]
        )
        mcp_tool_slots = sum(
            str(slot["phase"]).startswith("mcp_tool_") for slot in cell["attempt_slots"]
        )
        input_token_bound = (
            Decimal(len(cell["task"]["prompt"].encode("utf-8")) + 2_000 + 24_000 + 65_536)
            / Decimal(3)
            * Decimal(5)
        )
        cohere_cell_limits[str(cell["work_item_id"])] = {
            "model_id": cell["model_id"],
            "task_id": cell["task_id"],
            "provider_attempt_slots": provider_attempt_slots,
            "semantic_successful_response_bound": 5,
            "mcp_session_slots": mcp_session_slots,
            "mcp_tool_call_slots": mcp_tool_slots,
            "max_actual_tool_calls": 12,
            "max_tool_rounds": 3,
            "max_cumulative_tool_result_bytes": 65_536,
            "max_intermediate_tokens": 8_192,
            "max_final_tokens": 8_192,
            "max_output_tokens_across_successful_responses": 40_960,
            "max_reasoning_tokens_across_successful_responses": 40_960,
            "max_input_tokens_across_successful_responses": _decimal_text(input_token_bound),
        }
    cohere_resource_envelope = {
        "schema_version": "flavourbench-cohere-prospective-resource-envelope-v1",
        "execution_policy_sha256": predecessor["primary_protocol"]["execution_policy_sha256"],
        "condition": "epicure_on",
        "provider": "cohere_direct",
        "credential_program": "Cohere Scholars",
        "usd_cost_or_reservation_claimed": False,
        "cell_limits": cohere_cell_limits,
        "totals": {
            "fresh_arms": len(cohere_cells),
            "provider_attempt_slots": sum(
                value["provider_attempt_slots"] for value in cohere_cell_limits.values()
            ),
            "semantic_successful_response_bound": sum(
                value["semantic_successful_response_bound"] for value in cohere_cell_limits.values()
            ),
            "mcp_session_slots": sum(
                value["mcp_session_slots"] for value in cohere_cell_limits.values()
            ),
            "mcp_tool_call_slots": sum(
                value["mcp_tool_call_slots"] for value in cohere_cell_limits.values()
            ),
            "max_actual_tool_calls": sum(
                value["max_actual_tool_calls"] for value in cohere_cell_limits.values()
            ),
            "max_output_tokens": sum(
                value["max_output_tokens_across_successful_responses"]
                for value in cohere_cell_limits.values()
            ),
            "max_reasoning_tokens": sum(
                value["max_reasoning_tokens_across_successful_responses"]
                for value in cohere_cell_limits.values()
            ),
            "max_input_tokens": _decimal_text(
                sum(
                    (
                        Decimal(value["max_input_tokens_across_successful_responses"])
                        for value in cohere_cell_limits.values()
                    ),
                    Decimal(0),
                )
            ),
        },
    }
    cohere_resource_envelope = {
        **cohere_resource_envelope,
        "envelope_sha256": sha256_json(cohere_resource_envelope),
    }
    historical_cohere = _historical_cohere_disclosure(repo_root)
    failed_v3 = _failed_v3_freeze_evidence(repo_root)
    retired_v4 = _retired_v4_format_freeze_evidence(repo_root)
    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_not_executed_independent_go_required",
        "freeze_nonce": FREEZE_NONCE,
        "supersedes_failed_offline_plan_sha256": FAILED_V3_PLAN_SHA256,
        "failed_offline_supersession": failed_v3,
        "supersedes_retired_format_only_v4_plan_sha256": RETIRED_V4_PLAN_SHA256,
        "retired_format_only_v4_freeze": retired_v4,
        "supersedes": {
            **_file_ref(repo_root, predecessor_path, expected_sha256=PREDECESSOR_SHA256),
            "reason": "repair_50_declared_arm_ids_and_bind_executor_source_and_exact_routes",
            "replay_from_predecessor_permitted": False,
        },
        "source_artifacts": {
            "predecessor_preflight": _file_ref(
                repo_root,
                repo_root / PREDECESSOR_PREFLIGHT,
                expected_sha256=PREDECESSOR_PREFLIGHT_SHA256,
            ),
            "offline_cohere_gate": _file_ref(
                repo_root, repo_root / COHERE_GATE, expected_sha256=COHERE_GATE_SHA256
            ),
            "corrected_arena": _file_ref(
                repo_root, arena_path, expected_sha256=CORRECTED_ARENA_SHA256
            ),
            "task_validity": _file_ref(repo_root, task_path, expected_sha256=TASK_VALIDITY_SHA256),
            "task_quarantine": _file_ref(
                repo_root, quarantine_path, expected_sha256=TASK_QUARANTINE_SHA256
            ),
            "route_manifests": manifest_refs,
        },
        "source_code": source_closure,
        "epicure": dict(epicure),
        "execution_policy": predecessor["primary_protocol"],
        "cohere_prospective_resource_envelope": cohere_resource_envelope,
        "cells": cells,
        "endpoint_batches": batches,
        "batch_execution_order": [batch["batch_id"] for batch in batches],
        "execution_roots": {
            "coordinator": (
                f"{DEFAULT_OUTPUT_ROOT}/run/coordinator"
            ),
            "canonical_global_source": "flavourbench/artifacts/live-smoke",
            "canonical_global_reservation_ledger": (
                "flavourbench/artifacts/frontier-contract/ledger.jsonl"
            ),
            "endpoints": {
                str(batch["batch_id"]): (
                    f"{DEFAULT_OUTPUT_ROOT}/run/endpoints/" + str(batch["batch_id"])
                )
                for batch in batches
            },
        },
        "identity_repair": {
            "fresh_cell_ids": 50,
            "fresh_work_ids": 50,
            "fresh_run_ids": 50,
            "fresh_arm_ids": 50,
            "fresh_attempt_ids": 1450,
            "fresh_batch_ids": 16,
            "new_identifiers_disjoint_from_all_prior_records": True,
            "prior_identifier_count": len(prior_identifiers),
            "prior_identifiers_sha256": sha256_json(sorted(prior_identifiers)),
            "prior_identifier_scan_roots": ["flavourbench/artifacts", "artifacts"],
            "prior_identifier_scan_fail_closed": True,
            "prior_identifier_snapshot_is_mutable_history_lock": False,
            "retired_predecessor_identifier_count": len(retired_identifiers),
            "retired_predecessor_identifiers_sha256": sha256_json(sorted(retired_identifiers)),
            "retired_failed_v3_identifier_count": failed_v3["retired_v3_identifier_count"],
            "retired_failed_v3_identifiers_sha256": failed_v3["retired_v3_identifiers_sha256"],
            "retired_format_only_v4_identifier_count": retired_v4[
                "retired_v4_identifier_count"
            ],
            "retired_format_only_v4_identifiers_sha256": retired_v4[
                "retired_v4_identifiers_sha256"
            ],
            "invariant": (
                "cell.arm_ids.epicure_on == run_id + ':epicure_on' == every attempt slot arm_id"
            ),
        },
        "support": {
            "observed_before": before,
            "observed_supported_model_pair_family_cells": 407,
            "observed_empty_model_pair_family_cells": 73,
            "conditional_after_all_50_usable": projected,
            "projection_is_observed": False,
            "projection_scope": "connectivity_only",
            "anchor_tasks": 4,
            "planned_arms_by_family": {
                family: sum(cell["task_family"] == family for cell in cells)
                for family in ("composition", "cookability", "evidence", "substitution")
            },
        },
        "budget": {
            "currency": "USD",
            "bound_predecessor_snapshot_exposure_usd": _decimal_text(baseline),
            "bound_predecessor_snapshot_derivation": baseline_derivation,
            "priced_routes_worst_case_usd": _decimal_text(priced_total),
            "priced_routes_projected_from_bound_snapshot_usd": _decimal_text(
                baseline + priced_total
            ),
            "unpriced_cohere_cells": 8,
            "complete_plan_reservation_bound": False,
            "cohere_current_usd_price_claimed": False,
            "cohere_current_usd_reservation_available": False,
            "cohere_resource_envelope_and_operator_attestation_required": True,
            "admission_ceiling_usd": _decimal_text(ADMISSION_CEILING_USD),
            "hard_cap_usd": _decimal_text(HARD_CAP_USD),
            "reservation_unit": "one_complete_endpoint_isolated_batch",
            "one_active_batch_at_a_time": True,
            "transactional_global_actual_and_active_reservation_rebase_before_each_batch": True,
            "bound_snapshot_projection_is_current_exposure_claim": False,
            "bound_snapshot_projection_is_durable_headroom": False,
            "cross_study_budget_contention_can_block_later_batches": True,
            "historical_source_provenance": historical_cohere,
            "admission_authorized": False,
        },
        "counts": {
            "cells": 50,
            "real_epicure_on_arms": 50,
            "synthetic_arms": 0,
            "attempt_slots": 1450,
            "endpoint_batches": 16,
            "openrouter_arms": 39,
            "kimi_direct_arms": 3,
            "cohere_direct_arms": 8,
            "provider_calls_by_freeze": 0,
            "catalog_calls_by_freeze": 0,
            "epicure_calls_by_freeze": 0,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_judgments": 0,
            "family_specific_ranking_supported": False,
            "permitted_analysis": "conditional_connectivity_and_reliability_diagnostics_only",
            "synthetic_arms": 0,
            "four_anchor_tasks_support_precision_claims": False,
        },
    }
    plan = {**payload, "artifact_sha256": _sha256(payload)}
    if owned_existing_plan is not None and owned_existing_plan != plan:
        raise CoverageSuccessorError(
            "existing successor v4 plan differs from the exact current build"
        )
    return plan


def validate_plan(
    plan: Mapping[str, Any], *, repo_root: Path, output_root: Path | None = None
) -> None:
    output_root = output_root or (repo_root / DEFAULT_OUTPUT_ROOT)
    body = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("artifact_sha256") != _sha256(body):
        raise CoverageSuccessorError("successor plan content address does not verify")
    verify_source_closure(expected=plan["source_code"], repo_root=repo_root)
    failed_v3 = _failed_v3_freeze_evidence(repo_root)
    retired_v4 = _retired_v4_format_freeze_evidence(repo_root)
    if (
        plan.get("status") != "frozen_not_executed_independent_go_required"
        or plan.get("supersedes_failed_offline_plan_sha256") != FAILED_V3_PLAN_SHA256
        or plan.get("failed_offline_supersession") != failed_v3
        or (plan.get("identity_repair") or {}).get("retired_failed_v3_identifier_count")
        != failed_v3["retired_v3_identifier_count"]
        or (plan.get("identity_repair") or {}).get("retired_failed_v3_identifiers_sha256")
        != failed_v3["retired_v3_identifiers_sha256"]
        or plan.get("supersedes_retired_format_only_v4_plan_sha256")
        != RETIRED_V4_PLAN_SHA256
        or plan.get("retired_format_only_v4_freeze") != retired_v4
        or (plan.get("identity_repair") or {}).get(
            "retired_format_only_v4_identifier_count"
        )
        != retired_v4["retired_v4_identifier_count"]
        or (plan.get("identity_repair") or {}).get(
            "retired_format_only_v4_identifiers_sha256"
        )
        != retired_v4["retired_v4_identifiers_sha256"]
    ):
        raise CoverageSuccessorError("failed v3 or retired v4 supersession evidence differs")
    for reference in [
        plan["supersedes"],
        *[value for key, value in plan["source_artifacts"].items() if key != "route_manifests"],
        *plan["source_artifacts"]["route_manifests"],
    ]:
        path = repo_root / str(reference["path"])
        if (
            path.stat().st_size != reference["bytes"]
            or _file_sha256(path) != reference["file_sha256"]
            or _file_ref(repo_root, path)["semantic_sha256"] != reference["semantic_sha256"]
        ):
            raise CoverageSuccessorError("successor input file identity differs")
    work: set[str] = set()
    runs: set[str] = set()
    arms: set[str] = set()
    attempts: set[str] = set()
    for cell in plan["cells"]:
        run_id = str(cell["run_id"])
        arm_id = str(cell["arm_ids"]["epicure_on"])
        cell_attempts = {str(slot["attempt_id"]) for slot in cell["attempt_slots"]}
        lineage = cell["predecessor_lineage"]
        identity_basis = {
            "schema_version": PLAN_SCHEMA,
            "freeze_nonce": FREEZE_NONCE,
            "predecessor_plan_sha256": PREDECESSOR_SHA256,
            "predecessor_cell_id": lineage["cell_id"],
            "ordinal": cell["ordinal"],
            "model_id": cell["model_id"],
            "task_id": cell["task_id"],
            "task_family": cell["task_family"],
            "route_manifest_sha256": cell["route_manifest_sha256"],
            "endpoint_execution_sha256": cell["endpoint_execution_sha256"],
        }
        expected_cell = sha256_json({**identity_basis, "identifier_role": "cell"})
        expected_work = sha256_json(
            {**identity_basis, "cell_id": expected_cell, "identifier_role": "work"}
        )
        expected_run = str(uuid.uuid5(NAMESPACE, f"{FREEZE_NONCE}:run:{expected_work}"))
        expected_slots = [
            {
                "arm_id": f"{expected_run}:epicure_on",
                "phase": str(slot["phase"]),
                "attempt_index": int(slot["attempt_index"]),
                "attempt_id": str(
                    uuid.uuid5(
                        NAMESPACE,
                        (
                            f"{FREEZE_NONCE}:attempt:{expected_work}:"
                            f"{expected_run}:epicure_on:{slot['phase']}:{slot['attempt_index']}"
                        ),
                    )
                ),
            }
            for slot in cell["attempt_slots"]
        ]
        if (
            cell["cell_id"] != expected_cell
            or cell["work_item_id"] != expected_work
            or run_id != expected_run
            or cell["attempt_slots"] != expected_slots
            or arm_id != f"{run_id}:epicure_on"
            or {str(slot["arm_id"]) for slot in cell["attempt_slots"]} != {arm_id}
            or len(cell_attempts) != 29
            or cell["work_item_id"] in work
            or run_id in runs
            or arm_id in arms
            or attempts.intersection(cell_attempts)
            or cell["route"]["provider_tag"] != cell["provider_tag"]
            or cell["route"]["execution_backend"] != cell["execution_backend"]
            or cell["route"]["endpoint_execution_sha256"] != cell["endpoint_execution_sha256"]
        ):
            raise CoverageSuccessorError("successor execution identity or route differs")
        work.add(str(cell["work_item_id"]))
        runs.add(run_id)
        arms.add(arm_id)
        attempts.update(cell_attempts)
    if (len(work), len(runs), len(arms), len(attempts)) != (50, 50, 50, 1450):
        raise CoverageSuccessorError("successor identity inventory is incomplete")
    cohere_envelope = plan.get("cohere_prospective_resource_envelope") or {}
    cohere_limits = cohere_envelope.get("cell_limits") or {}
    cohere_cells = [cell for cell in plan["cells"] if cell["execution_backend"] == "cohere_direct"]
    if (
        not isinstance(cohere_limits, Mapping)
        or set(map(str, cohere_limits)) != {str(cell["work_item_id"]) for cell in cohere_cells}
        or cohere_envelope.get("usd_cost_or_reservation_claimed") is not False
        or cohere_envelope.get("execution_policy_sha256")
        != plan["execution_policy"]["execution_policy_sha256"]
        or cohere_envelope.get("envelope_sha256")
        != sha256_json(
            {key: value for key, value in cohere_envelope.items() if key != "envelope_sha256"}
        )
        or cohere_envelope.get("totals")
        != {
            "fresh_arms": 8,
            "provider_attempt_slots": 80,
            "semantic_successful_response_bound": 40,
            "mcp_session_slots": 8,
            "mcp_tool_call_slots": 144,
            "max_actual_tool_calls": 96,
            "max_output_tokens": 327_680,
            "max_reasoning_tokens": 327_680,
            "max_input_tokens": _decimal_text(
                sum(
                    (
                        (
                            Decimal(
                                len(cell["task"]["prompt"].encode("utf-8"))
                                + 2_000
                                + 24_000
                                + 65_536
                            )
                            / Decimal(3)
                            * Decimal(5)
                        )
                        for cell in cohere_cells
                    ),
                    Decimal(0),
                )
            ),
        }
    ):
        raise CoverageSuccessorError("prospective Cohere resource envelope does not rederive")
    priced_cells = [cell for cell in plan["cells"] if cell["execution_backend"] != "cohere_direct"]
    expected_cohere_status = NON_USD_UNKNOWN_STATUS
    if (
        len(cohere_cells) != 8
        or len(priced_cells) != 42
        or any(
            set(cell["cost_reservation"])
            != {
                "status",
                "currency",
                "successor_reservation_usd",
                "current_usd_price_or_reservation_available",
                "historical_source_provenance",
            }
            or cell["cost_reservation"].get("status") != expected_cohere_status
            or cell["cost_reservation"].get("currency") is not None
            or cell["cost_reservation"].get("successor_reservation_usd") is not None
            or cell["cost_reservation"].get("current_usd_price_or_reservation_available")
            is not False
            or "reserved_worst_case_usd" in cell
            or (cell.get("route") or {}).get("pricing_status") != expected_cohere_status
            or (cell.get("route") or {}).get("pricing_currency") is not None
            or set(((cell.get("route") or {}).get("pricing") or {}).values()) != {None}
            or not isinstance(
                (cell["cost_reservation"] or {}).get("historical_source_provenance"),
                Mapping,
            )
            or not isinstance(
                (cell.get("route") or {}).get("historical_source_provenance"), Mapping
            )
            for cell in cohere_cells
        )
        or any(
            cell["cost_reservation"].get("status") != "priced_frozen_route"
            or cell["cost_reservation"].get("successor_reservation_usd") is None
            or _decimal(
                cell["cost_reservation"]["successor_reservation_usd"],
                field="priced successor cell reservation",
            )
            <= 0
            or cell["cost_reservation"].get("current_price_claimed") is not True
            for cell in priced_cells
        )
        or _current_cohere_economic_ambiguities(plan)
    ):
        raise CoverageSuccessorError("successor cell cost semantics differ")
    cells_by_work = {str(cell["work_item_id"]): cell for cell in plan["cells"]}
    priced_total = Decimal(0)
    batch_ids: set[str] = set()
    for batch in plan["endpoint_batches"]:
        expected_batch = sha256_json(
            {
                "schema_version": PLAN_SCHEMA,
                "freeze_nonce": FREEZE_NONCE,
                "identifier_role": "endpoint_batch",
                "predecessor_batch_id": batch["predecessor_batch_id"],
                "execution_ordinal": batch["execution_ordinal"],
                "model_id": batch["model_id"],
                "work_item_ids": batch["work_item_ids"],
            }
        )
        batch_cells = [cells_by_work[str(work_id)] for work_id in batch["work_item_ids"]]
        unpriced = [
            cell
            for cell in batch_cells
            if cell["cost_reservation"]["successor_reservation_usd"] is None
        ]
        priced = sum(
            (
                _decimal(
                    cell["cost_reservation"]["successor_reservation_usd"],
                    field="priced successor batch cell reservation",
                )
                for cell in batch_cells
                if cell["cost_reservation"]["successor_reservation_usd"] is not None
            ),
            Decimal(0),
        )
        if batch["batch_id"] != expected_batch or expected_batch in batch_ids:
            raise CoverageSuccessorError("successor batch identity does not rederive")
        if batch["execution_backend"] == "cohere_direct":
            if (
                len(unpriced) != len(batch_cells)
                or batch.get("unpriced_cell_count") != len(batch_cells)
                or batch.get("complete_reservation_bound") is not False
                or batch.get("successor_priced_reserve_usd") is not None
                or "predecessor_worst_case_usd" in batch
                or not isinstance(batch.get("historical_source_provenance"), Mapping)
            ):
                raise CoverageSuccessorError(
                    "Cohere batch is not preserved as non-USD reservation unknown"
                )
        else:
            if (
                unpriced
                or batch.get("unpriced_cell_count") != 0
                or batch.get("complete_reservation_bound") is not True
                or _decimal(
                    batch.get("successor_priced_reserve_usd"),
                    field="priced successor batch reservation",
                )
                != priced
            ):
                raise CoverageSuccessorError("priced successor batch reservation differs")
            priced_total += priced
        batch_ids.add(expected_batch)
    budget = plan.get("budget") or {}
    if (
        budget.get("currency") != "USD"
        or budget.get("unpriced_cohere_cells") != 8
        or budget.get("complete_plan_reservation_bound") is not False
        or budget.get("cohere_current_usd_price_claimed") is not False
        or budget.get("cohere_current_usd_reservation_available") is not False
        or budget.get("cohere_resource_envelope_and_operator_attestation_required") is not True
        or _decimal(
            budget.get("priced_routes_worst_case_usd"),
            field="successor priced route total",
        )
        != priced_total
        or _decimal(
            budget.get("priced_routes_projected_from_bound_snapshot_usd"),
            field="successor priced projection",
        )
        != _decimal(
            budget.get("bound_predecessor_snapshot_exposure_usd"),
            field="bound predecessor snapshot",
        )
        + priced_total
    ):
        raise CoverageSuccessorError("successor plan budget semantics differ")
    owned_output_roots, owned_existing_plan = _discover_existing_v4_offline_outputs(
        repo_root=repo_root,
        output_root=output_root,
        existing_output=None,
    )
    if owned_existing_plan is not None and owned_existing_plan != plan:
        raise CoverageSuccessorError("existing successor v4 plan differs from validation input")
    current_prior: set[str] = set()
    for governed_root in (repo_root / "flavourbench/artifacts", repo_root / "artifacts"):
        if governed_root.exists():
            current_prior.update(
                _prior_identifiers(
                    governed_root,
                    verified_successor_output=output_root,
                    successor_plan=plan,
                    additional_verified_successor_outputs=owned_output_roots,
                )
            )
    inventory = plan["identity_repair"]
    all_new = (
        work | runs | arms | attempts | batch_ids | {str(cell["cell_id"]) for cell in plan["cells"]}
    )
    if current_prior.intersection(all_new):
        raise CoverageSuccessorError("successor IDs are not disjoint from the prior inventory")
    if (
        not isinstance(inventory.get("prior_identifier_count"), int)
        or not isinstance(inventory.get("prior_identifiers_sha256"), str)
        or inventory.get("prior_identifier_snapshot_is_mutable_history_lock") is not False
    ):
        raise CoverageSuccessorError("frozen prior-identifier provenance is malformed")
    if plan["claim_boundary"] != {
        "development_only": True,
        "official": False,
        "rank_eligible": False,
        "quality_judgments": 0,
        "family_specific_ranking_supported": False,
        "permitted_analysis": "conditional_connectivity_and_reliability_diagnostics_only",
        "synthetic_arms": 0,
        "four_anchor_tasks_support_precision_claims": False,
    }:
        raise CoverageSuccessorError("successor claim boundary changed")
    if (
        plan.get("execution_roots", {}).get("canonical_global_source")
        != "flavourbench/artifacts/live-smoke"
        or plan.get("execution_roots", {}).get("canonical_global_reservation_ledger")
        != "flavourbench/artifacts/frontier-contract/ledger.jsonl"
        or plan.get("support", {}).get("observed_supported_model_pair_family_cells") != 407
        or plan.get("support", {}).get("observed_empty_model_pair_family_cells") != 73
        or plan.get("support", {}).get("projection_is_observed") is not False
        or plan.get("budget", {}).get("historical_source_provenance")
        != _historical_cohere_disclosure(repo_root)
    ):
        raise CoverageSuccessorError("successor source or observed-support boundary changed")


def build_preflight(
    *, plan: Mapping[str, Any], repo_root: Path, output_root: Path | None = None
) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root, output_root=output_root)
    roots = [
        repo_root / str(plan["execution_roots"]["coordinator"]),
        *(repo_root / str(value) for value in plan["execution_roots"]["endpoints"].values()),
    ]
    existing = sorted(_relative(repo_root, root) for root in roots if root.exists())
    blockers = [
        {
            "code": "cohere_complete_reservation_envelope_missing",
            "scope": "cohere_batches_only_8_direct_cells",
            "reason": (
                "the exact non-USD resource envelope is frozen, but an actual secret-free "
                "operator attestation supplying economic authorization for that bounded "
                "Scholars use is absent; no USD reservation or zero-price claim exists"
            ),
        },
        {
            "code": "live_route_availability_not_tested",
            "scope": "all_16_exact_routes",
            "reason": "offline manifest verification is not a live availability attestation",
        },
        {
            "code": "reasoning_effort_sensitivity_precedes_coverage",
            "scope": "study_order",
            "reason": "coverage collection must remain quiescent until sensitivity review closes",
        },
        {
            "code": "cross_study_budget_contention_requires_locked_rebase",
            "scope": "shared_85_usd_admission_ceiling",
            "reason": (
                "the predecessor snapshot plus all priced coverage fits only before later "
                "runs; each batch requires current actual and active-reservation accounting"
            ),
        },
        {
            "code": "epicure_lineage_not_independently_reconstructable",
            "scope": "release_claim",
            "reason": "the bound unmatched runtime remains explanatory development evidence only",
        },
        {
            "code": "independent_governance_go_required",
            "scope": "paid_execution",
            "reason": "this offline freeze grants no collection authority",
        },
    ]
    if existing:
        blockers.insert(
            0,
            {
                "code": "successor_execution_root_not_empty",
                "scope": existing,
                "reason": "a fresh successor requires absent execution roots",
            },
        )
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "frozen_blocked_zero_call_preflight",
        "plan_sha256": plan["artifact_sha256"],
        "source_closure_sha256": plan["source_code"]["closure_sha256"],
        "environment_sha256": plan["source_code"]["execution_environment"]["environment_sha256"],
        "checks": {
            "all_50_arm_id_invariants_pass": True,
            "all_50_work_run_identities_fresh_and_rederived": True,
            "all_1450_attempt_ids_fresh_and_rederived": True,
            "all_16_routes_offline_content_address_verified": True,
            "live_route_availability_verified": False,
            "execution_roots_absent": not existing,
            "priced_route_budget_mathematically_fits": (
                Decimal(plan["budget"]["priced_routes_projected_from_bound_snapshot_usd"])
                <= ADMISSION_CEILING_USD
            ),
            "bound_snapshot_is_current_exposure_claim": False,
            "complete_plan_reservation_bound": False,
            "cohere_current_usd_price_claimed": False,
            "cohere_current_usd_reservation_available": False,
        },
        "budget": dict(plan["budget"]),
        "support": dict(plan["support"]),
        "blockers": blockers,
        "decision": "execution_not_admitted",
        "calls_made": {"provider_completions": 0, "catalog_gets": 0, "epicure": 0},
        "claim_boundary": dict(plan["claim_boundary"]),
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def verify_preflight(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    repo_root: Path,
    output_root: Path | None = None,
) -> None:
    validate_plan(plan, repo_root=repo_root, output_root=output_root)
    body = {key: value for key, value in preflight.items() if key != "artifact_sha256"}
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA
        or preflight.get("artifact_sha256") != _sha256(body)
        or preflight.get("plan_sha256") != plan["artifact_sha256"]
        or preflight.get("decision") != "execution_not_admitted"
        or not preflight.get("blockers")
        or preflight.get("calls_made")
        != {"provider_completions": 0, "catalog_gets": 0, "epicure": 0}
    ):
        raise CoverageSuccessorError("successor preflight does not verify")


def build_dry_run(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    repo_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    verify_preflight(plan=plan, preflight=preflight, repo_root=repo_root, output_root=output_root)
    decisions = []
    for batch in plan["endpoint_batches"]:
        decisions.append(
            {
                "execution_ordinal": batch["execution_ordinal"],
                "batch_id": batch["batch_id"],
                "model_id": batch["model_id"],
                "execution_backend": batch["execution_backend"],
                "provider_tag": batch["provider_tag"],
                "cell_count": batch["cell_count"],
                "complete_reservation_bound": batch["complete_reservation_bound"],
                "decision": (
                    "blocked_cohere_resource_envelope_and_operator_attestation"
                    if not batch["complete_reservation_bound"]
                    else "blocked_by_zero_call_preflight"
                ),
            }
        )
    payload = {
        "schema_version": DRY_RUN_SCHEMA,
        "status": "blocked_dry_run_no_calls",
        "plan_sha256": plan["artifact_sha256"],
        "preflight_sha256": preflight["artifact_sha256"],
        "executor_entrypoint": "flavourbench.frontier_coverage_primary_executor_v1",
        "source_closure_sha256": plan["source_code"]["closure_sha256"],
        "decisions": decisions,
        "counts": {
            "batches_considered": 16,
            "cells_considered": 50,
            "provider_completions": 0,
            "catalog_gets": 0,
            "epicure_calls": 0,
            "ledger_events": 0,
            "synthetic_arms": 0,
        },
        "support": dict(plan["support"]),
        "budget": dict(plan["budget"]),
        "blockers": list(preflight["blockers"]),
        "claim_boundary": dict(plan["claim_boundary"]),
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def build_cohere_operator_attestation_template(*, plan: Mapping[str, Any]) -> dict[str, Any]:
    envelope = plan["cohere_prospective_resource_envelope"]
    work_item_ids = _ordered_cohere_work_item_ids(plan)
    if set(work_item_ids) != set(envelope["cell_limits"]):
        raise CoverageSuccessorError("ordered Cohere work IDs differ from the resource envelope")
    payload = {
        "schema_version": COHERE_OPERATOR_TEMPLATE_SCHEMA,
        "status": "template_not_authorization",
        "plan_sha256": plan["artifact_sha256"],
        "provider": "cohere_direct",
        "credential_program": "Cohere Scholars",
        "work_item_ids": work_item_ids,
        "resource_envelope_sha256": envelope["envelope_sha256"],
        "resource_envelope_totals": dict(envelope["totals"]),
        "execution_policy_sha256": plan["execution_policy"]["execution_policy_sha256"],
        "required_attestation_fields": {
            "status": "operator_authorized_exact_resource_scope",
            "decision": "authorize_exact_bounded_cohere_scholars_use",
            "decision_semantics": (
                "required_non_usd_economic_authorization_for_exact_resource_scope_not_a_"
                "zero_price_claim"
            ),
            "operator": {"full_name": "required", "role": "required"},
            "issued_at": "required_RFC3339_timezone_aware",
            "expires_at": "required_RFC3339_timezone_aware_and_future",
            "credential_binding_method": "sha256_canonical_public_binding_object",
            "credential_binding_is_derived_from_secret": False,
            "provider_or_epicure_calls_made_by_attestation": False,
        },
        "credential_binding_rule": {
            "digest_input": "public_binding_object_only",
            "public_binding_object": {
                "provider": "cohere_direct",
                "credential_program": "Cohere Scholars",
                "environment_variable_name": "COHERE_API_KEY",
                "credential_handle": "operator_selected_nonsecret_handle",
                "scope": {
                    "plan_sha256": plan["artifact_sha256"],
                    "resource_envelope_sha256": envelope["envelope_sha256"],
                    "work_item_ids_sha256": _sha256(work_item_ids),
                    "authorized_use": "frontier_coverage_successor_cohere_direct_only",
                },
            },
            "digest_algorithm": "sha256_canonical_json",
            "must_not_hash_or_fingerprint_credential_material": True,
        },
        "prohibitions": {
            "credential_or_secret_values": True,
            "credential_hash_or_fingerprint": True,
            "usd_price_cost_reservation_or_release_claim": True,
            "authorization_outside_exact_work_item_ids": True,
        },
        "contains_secret": False,
        "usd_cost_or_reservation_claimed": False,
    }
    if _contains_secret_material(payload):
        raise CoverageSuccessorError("Cohere operator template contains secret material")
    return {**payload, "artifact_sha256": _sha256(payload)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("command", choices=["freeze"])
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    try:
        existing_output = _verify_existing_offline_output_root(
            output_root=output, repo_root=repo_root
        )
        plan = build_plan(repo_root=repo_root, output_root=output)
        preflight = build_preflight(plan=plan, repo_root=repo_root, output_root=output)
        dry_run = build_dry_run(
            plan=plan, preflight=preflight, repo_root=repo_root, output_root=output
        )
        cohere_template = build_cohere_operator_attestation_template(plan=plan)
        generated = {
            "plan": plan,
            "preflight": preflight,
            "dry-run": dry_run,
            "templates": cohere_template,
        }
        if existing_output is not None and existing_output != generated:
            raise CoverageSuccessorError(
                "existing successor v4 artifact set differs from the exact current build"
            )
        plan_path = _write_artifact(
            output / "plan", "frontier-coverage-primary-successor-plan", plan
        )
        preflight_path = _write_artifact(
            output / "preflight", "frontier-coverage-primary-successor-preflight", preflight
        )
        dry_run_path = _write_artifact(
            output / "dry-run", "frontier-coverage-primary-successor-dry-run", dry_run
        )
        cohere_template_path = _write_artifact(
            output / "templates",
            "cohere-scholars-operator-attestation-template",
            cohere_template,
        )
    except (CoverageSuccessorError, IntegrityError, SourceClosureError) as error:
        raise SystemExit(f"offline successor freeze failed: {error}") from error
    print(
        json.dumps(
            {
                "status": dry_run["status"],
                "plan": str(plan_path),
                "plan_sha256": plan["artifact_sha256"],
                "preflight": str(preflight_path),
                "preflight_sha256": preflight["artifact_sha256"],
                "dry_run": str(dry_run_path),
                "dry_run_sha256": dry_run["artifact_sha256"],
                "cohere_operator_attestation_template": str(cohere_template_path),
                "cohere_operator_attestation_template_sha256": cohere_template["artifact_sha256"],
                "provider_completions": 0,
                "catalog_gets": 0,
                "epicure_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
