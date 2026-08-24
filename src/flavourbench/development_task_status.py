"""Verify and seal the privacy-safe live task-validation status artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-development-task-validation-status-v1"
STATISTICS_SCHEMA_VERSION = "flavourbench-development-task-validation-statistics-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROHIBITED_KEYS = {
    "reviewerId",
    "reviewer_id",
    "identityCommitmentSha256",
    "identity_commitment_sha256",
    "invitation",
    "note",
    "prompt",
    "humanReference",
    "sourceAuthor",
}


class DevelopmentTaskStatusError(ValueError):
    """The status artifact is inconsistent, unsafe, or not content addressed."""


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def verify_status_artifact(
    document: Mapping[str, Any],
    *,
    expected_packet_sha256: str | None = None,
) -> None:
    payload = {key: value for key, value in document.items() if key != "artifactSha256"}
    digest = str(document.get("artifactSha256") or "")
    if not _SHA256.fullmatch(digest) or sha256_json(payload) != digest:
        raise DevelopmentTaskStatusError("status artifact content address does not verify")
    if document.get("schemaVersion") != SCHEMA_VERSION:
        raise DevelopmentTaskStatusError("unexpected status artifact schema")
    packet_sha256 = str(document.get("packetSha256") or "")
    if not _SHA256.fullmatch(packet_sha256):
        raise DevelopmentTaskStatusError("status artifact has no packet binding")
    if expected_packet_sha256 is not None and packet_sha256 != expected_packet_sha256:
        raise DevelopmentTaskStatusError("status artifact binds the wrong packet")
    if _PROHIBITED_KEYS & _walk_keys(document):
        raise DevelopmentTaskStatusError("status artifact contains prohibited review material")

    task_count = int(document.get("taskCount", -1))
    required_per_task = int(document.get("requiredIndependentReviewsPerTask", -1))
    tasks = document.get("tasks")
    if task_count != 40 or required_per_task != 3:
        raise DevelopmentTaskStatusError("status artifact has the wrong campaign denominator")
    if not isinstance(tasks, list) or len(tasks) != task_count:
        raise DevelopmentTaskStatusError("status artifact task rows are incomplete")
    task_ids = [str(task.get("taskId") or "") for task in tasks if isinstance(task, Mapping)]
    if len(task_ids) != task_count or len(set(task_ids)) != task_count or not all(task_ids):
        raise DevelopmentTaskStatusError("status artifact task IDs are not unique")

    statistics = document.get("statistics")
    if not isinstance(statistics, Mapping) or (
        statistics.get("schemaVersion") != STATISTICS_SCHEMA_VERSION
    ):
        raise DevelopmentTaskStatusError("status artifact has no verified statistics payload")
    coverage = statistics.get("coverage")
    if not isinstance(coverage, Mapping):
        raise DevelopmentTaskStatusError("status artifact has no coverage statistics")
    if (
        coverage.get("tasks") != task_count
        or coverage.get("requiredIndependentReviews") != task_count * required_per_task
        or coverage.get("completeIndependentReviews")
        != document.get("completeIndependentReviews")
        or coverage.get("criterionPacks") != document.get("humanCriterionPacks")
    ):
        raise DevelopmentTaskStatusError("status coverage disagrees with the campaign totals")
    boundary = statistics.get("claimBoundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("realSealedHumanRecordsOnly") is not True
        or boundary.get("missingReviewsImputed") is not False
        or boundary.get("packetRowsCountAsHumanEvidence") is not False
        or boundary.get("descriptiveNotConfirmatory") is not True
    ):
        raise DevelopmentTaskStatusError("status artifact crosses its evidence boundary")
    agreement = statistics.get("agreement")
    if not isinstance(agreement, Mapping):
        raise DevelopmentTaskStatusError("status artifact has no agreement statistics")
    if int(agreement.get("taskCount", -1)) == 0 and any(
        agreement.get(metric) is not None
        for metric in ("unanimousDecisionRate", "meanPairwiseAgreement", "fleissKappa")
    ):
        raise DevelopmentTaskStatusError("undefined agreement statistics must remain null")


def _load(path: str) -> dict[str, Any]:
    try:
        if path == "-":
            value = json.load(sys.stdin)
        else:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentTaskStatusError("status input is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DevelopmentTaskStatusError("status input must be a JSON object")
    return value


def write_status_artifact(
    document: Mapping[str, Any],
    *,
    output_dir: Path,
    expected_packet_sha256: str | None = None,
) -> Path:
    verify_status_artifact(document, expected_packet_sha256=expected_packet_sha256)
    digest = str(document["artifactSha256"])
    rendered = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"development-task-validation-status-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise DevelopmentTaskStatusError("content-addressed status output conflicts")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="-")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256")
    arguments = parser.parse_args(argv)
    path = write_status_artifact(
        _load(arguments.input),
        output_dir=arguments.output_dir,
        expected_packet_sha256=arguments.expected_packet_sha256,
    )
    print(path)


if __name__ == "__main__":
    run()
