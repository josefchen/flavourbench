"""Captured contamination-method successor for task-validation campaign v6.

This module is intentionally isolated from the service runtime.  It captures a
versioned benchmark corpus and one external web-search response for every frozen
task, then replays the campaign's exact, fuzzy, n-gram, semantic, and web methods
over those bytes.  Collection and replay are separate operations: live network
responses are never consulted while rebuilding the final artifact.

The successor cannot manufacture the independently labelled calibration set
required by the frozen Season 1 design.  Unless a real calibration receipt and
an independently adequate external-corpus inventory exist, its only valid
release disposition is ``no_go``.
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import html
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .prospective_task_acquisition import canonical_json_bytes, canonical_sha256
from .task_evidence import (
    CONTAMINATION_AUTOREJECT_THRESHOLDS,
    CONTAMINATION_REPORT_THRESHOLDS,
    normalized_prompt_sha256,
)
from .task_validation_automated_replay import (
    PINNED_INPUTS,
    PINNED_REPLAY_PHYSICAL_SHA256,
    PINNED_REPLAY_SEMANTIC_SHA256,
    ReplayInputPaths,
    build_replay_artifact,
)

BENCHMARK_SNAPSHOT_SCHEMA = "flavourbench-contamination-benchmark-snapshot-v2"
WEB_SNAPSHOT_SCHEMA = "flavourbench-contamination-web-snapshot-v2"
REPLAY_SCHEMA = "flavourbench-task-validation-contamination-replay-v2"
POLICY_VERSION = "flavourbench-task-validation-contamination-policy-v2"
ARTIFACT_ROLE = "captured_contamination_method_no_go_successor"
ASSIGNED_PROMPT_COUNT = 180
RAW_BENCHMARK_DIRECTORY = "raw-benchmark"
RAW_WEB_DIRECTORY = "raw-web"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

# Filled only after a capture is independently rebuilt.  They are verifier
# pins, never inputs to semantic hashing.
PINNED_BENCHMARK_SNAPSHOT_SEMANTIC_SHA256 = (
    "79133f28e52dd4c7bdbef74600c27c1907d0b2d6085157742a41a8c59491acfe"
)
PINNED_BENCHMARK_SNAPSHOT_PHYSICAL_SHA256 = (
    "43c75af75e0c6b09bdd8c982f1a3bc061a64efbf81c17f16024cbddc03cc5c42"
)
PINNED_WEB_SNAPSHOT_SEMANTIC_SHA256 = (
    "6dcb3a70783eea69cc247898baafb4f69c4dd029353aeac7e168a986714119d5"
)
PINNED_WEB_SNAPSHOT_PHYSICAL_SHA256 = (
    "346492aa23274484f0afe2f9590ac010b2ec17e9de717d344c77b4acf9859db3"
)
PINNED_REPLAY_V2_SEMANTIC_SHA256 = (
    "2c7e2ead2e4e936e840d6b0fbc9bbf268c3237e908da903982a00a1af0f0b44d"
)
PINNED_REPLAY_V2_PHYSICAL_SHA256 = (
    "9e520aa4f2384779efafb930d20cebe8f72b728197538adb43405943674abdb0"
)

HTTP_USER_AGENT = (
    "FlavourBenchContaminationAudit/2.0 (+https://epicure.kaikaku.ai/flavourbench/methodology)"
)
HF_DATASET_API = "https://huggingface.co/api/datasets/{dataset}"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
BING_RSS_ENDPOINT = "https://www.bing.com/search"


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    dataset: str
    revision: str
    config: str
    split: str
    expected_rows: int
    license_id: str
    license_url: str
    scope_reason: str


BENCHMARK_DATASETS: tuple[BenchmarkDatasetSpec, ...] = (
    BenchmarkDatasetSpec(
        dataset="cais/mmlu",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        config="nutrition",
        split="test",
        expected_rows=306,
        license_id="MIT",
        license_url="https://opensource.org/license/mit",
        scope_reason="food-adjacent subject in a widely used language-model benchmark",
    ),
    BenchmarkDatasetSpec(
        dataset="truthfulqa/truthful_qa",
        revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        config="generation",
        split="validation",
        expected_rows=817,
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        scope_reason="public question-answer benchmark with food and misconception items",
    ),
    BenchmarkDatasetSpec(
        dataset="allenai/ai2_arc",
        revision="210d026faf9955653af8916fad021475a3f00453",
        config="ARC-Challenge",
        split="test",
        expected_rows=1172,
        license_id="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        scope_reason="public science and commonsense benchmark test questions",
    ),
    BenchmarkDatasetSpec(
        dataset="allenai/ai2_arc",
        revision="210d026faf9955653af8916fad021475a3f00453",
        config="ARC-Easy",
        split="test",
        expected_rows=2376,
        license_id="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        scope_reason="public science and commonsense benchmark test questions",
    ),
)

# Culinary datasets that cannot safely be represented as scanned content in
# this release.  Their metadata is captured, but their records are excluded.
EXCLUDED_CULINARY_DATASETS: tuple[dict[str, str], ...] = (
    {
        "dataset": "lyan62/FoodieQA",
        "expected_revision": "a350780c299f8cf5a21c9665b7bd8b4665386675",
        "source_url": "https://huggingface.co/datasets/lyan62/FoodieQA",
        "reason": "gated evaluation-only access was not accepted by this automated capture",
        "license_status": "CC-BY-NC-ND-4.0 plus repository-specific access conditions",
    },
    {
        "dataset": "AdaptLLM/food-VQA-benchmark",
        "expected_revision": "9f2ac8108121d9245107b5ac6366cd59a23b9953",
        "source_url": "https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark",
        "reason": "dataset metadata does not declare a content license",
        "license_status": "unknown",
    },
)

WEB_PROVIDER_CONTRACT: dict[str, Any] = {
    "provider": "Bing web search RSS",
    "endpoint": BING_RSS_ENDPOINT,
    "interface": "public RSS response selected with format=rss",
    "official_authenticated_api": False,
    "query_policy": "one quoted exact title query per assigned prompt; ten requested results",
    "terms_url": "https://www.microsoft.com/en-us/servicesagreement",
    "result_text_redistribution_license": None,
    "result_text_license_status": "unknown",
    "known_positive_requirement": (
        "the already-public attributed source page should be returned for every exact-title query"
    ),
}
WEB_PROVIDER_CONTRACT_SHA256 = canonical_sha256(WEB_PROVIDER_CONTRACT)

BENCHMARK_SCAN_IMPLEMENTATION_VERSION = "flavourbench-task-validation-captured-benchmark-scan-v2"
BENCHMARK_SCAN_POLICY: dict[str, Any] = {
    "text_normalization": "NFKC, casefold, whitespace collapse",
    "exact": "normalized equality or >=40-character containment in either direction",
    "fuzzy": "normalized character-four-gram Sorensen-Dice",
    "ngram": "non-stopword token trigram Jaccard",
    "semantic": "distributional-random-indexing-v1, 64 dimensions, two-token context",
    "semantic_random_seed": "sha256('flavourbench-ri-v1\\0' + token)",
    "report_thresholds": {
        method: CONTAMINATION_REPORT_THRESHOLDS[method]
        for method in ("exact", "fuzzy", "ngram", "semantic")
    },
    "score_rounding": "nearest integer thousandth, ties to Python round semantics",
}
BENCHMARK_SCAN_IMPLEMENTATION_SHA256 = canonical_sha256(BENCHMARK_SCAN_POLICY)


class ContaminationReplayV2Error(ValueError):
    """A capture, content address, or fail-closed invariant did not verify."""


@dataclass(frozen=True)
class ReplayV2Paths:
    root: Path
    v1_inputs: ReplayInputPaths
    benchmark_snapshot: Path
    web_snapshot: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        benchmark_snapshot: Path,
        web_snapshot: Path,
    ) -> ReplayV2Paths:
        return cls(
            root=root,
            v1_inputs=ReplayInputPaths.from_root(root),
            benchmark_snapshot=benchmark_snapshot,
            web_snapshot=web_snapshot,
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _physical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )


def _seal_document(document: dict[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in document:
        raise ContaminationReplayV2Error("artifact payload is already sealed")
    sealed = dict(document)
    sealed["artifact_sha256"] = canonical_sha256(document)
    return sealed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_artifact(document: dict[str, Any], directory: Path, prefix: str) -> Path:
    digest = str(document.get("artifact_sha256", ""))
    if not SHA256_RE.fullmatch(digest) or digest != _artifact_digest(document):
        raise ContaminationReplayV2Error("cannot write an unsealed artifact")
    path = directory / f"{prefix}-{digest}.json"
    payload = canonical_json_bytes(document) + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ContaminationReplayV2Error("content-addressed destination already differs")
    else:
        _atomic_write(path, payload)
    return path


def _parse_http_date(value: str | None) -> str:
    if not value:
        raise ContaminationReplayV2Error("external response omitted the HTTP Date header")
    parsed = email.utils.parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _request_bytes(url: str, *, timeout: float = 45.0) -> tuple[bytes, dict[str, str], int]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json, application/rss+xml"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        headers = {key.casefold(): value for key, value in response.headers.items()}
        return body, headers, int(response.status)


def _safe_raw_path(snapshot_directory: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ContaminationReplayV2Error("raw capture path escapes the snapshot directory")
    resolved = snapshot_directory.joinpath(*pure.parts)
    if resolved.is_symlink() or not resolved.is_file():
        raise ContaminationReplayV2Error("raw capture is missing or symlinked")
    return resolved


def _capture_raw(
    directory: Path,
    *,
    subdirectory: str,
    suffix: str,
    body: bytes,
) -> tuple[str, str]:
    digest = _sha256_bytes(body)
    relative = f"{subdirectory}/{digest}.{suffix}"
    destination = directory / relative
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != body:
            raise ContaminationReplayV2Error("raw content-addressed capture differs")
    else:
        _atomic_write(destination, body)
    return relative, digest


def _load_assignment_rows(root: Path) -> list[dict[str, Any]]:
    paths = ReplayInputPaths.from_root(root)
    # Rebuilding v1 verifies all six pinned inputs and their cross-links before
    # the successor trusts the assignment payload.
    build_replay_artifact(paths)
    document = json.loads(paths.review_assignment.read_text(encoding="utf-8"))
    rows = document.get("assignment_rows")
    if not isinstance(rows, list) or len(rows) != ASSIGNED_PROMPT_COUNT:
        raise ContaminationReplayV2Error("frozen review assignment is not the 180-row slate")
    if len({str(row.get("candidate_id")) for row in rows}) != ASSIGNED_PROMPT_COUNT:
        raise ContaminationReplayV2Error("frozen review assignment contains duplicate candidates")
    ordered = sorted(rows, key=lambda row: int(row["assignment_ordinal"]))
    if [int(row["assignment_ordinal"]) for row in ordered] != list(
        range(1, ASSIGNED_PROMPT_COUNT + 1)
    ):
        raise ContaminationReplayV2Error("assignment ordinals are not contiguous")
    for row in ordered:
        prompt = row.get("prompt")
        source = row.get("source_metadata_visible_after_blind_decision")
        if not isinstance(prompt, str) or not isinstance(source, Mapping):
            raise ContaminationReplayV2Error("assignment prompt or source metadata is absent")
        if _sha256_bytes(prompt.encode("utf-8")) != row.get("prompt_sha256"):
            raise ContaminationReplayV2Error("assignment prompt digest drifted")
    return ordered


def _hf_metadata_license(metadata: Mapping[str, Any]) -> set[str]:
    card_data = metadata.get("cardData")
    value = card_data.get("license") if isinstance(card_data, Mapping) else None
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.casefold()}
    if isinstance(value, list):
        return {str(item).casefold() for item in value}
    raise ContaminationReplayV2Error("Hugging Face license metadata has an unexpected shape")


def _expected_hf_license(spec: BenchmarkDatasetSpec) -> str:
    return {
        "MIT": "mit",
        "Apache-2.0": "apache-2.0",
        "CC-BY-SA-4.0": "cc-by-sa-4.0",
    }[spec.license_id]


def capture_benchmark_snapshot(root: Path, output_directory: Path) -> Path:
    """Capture pinned, licensed public benchmark questions and exclusion receipts."""

    _load_assignment_rows(root)
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_receipts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    metadata_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for spec in BENCHMARK_DATASETS:
        if spec.dataset not in metadata_cache:
            metadata_url = HF_DATASET_API.format(dataset=spec.dataset)
            body, headers, status = _request_bytes(metadata_url)
            raw_path, raw_sha = _capture_raw(
                output_directory,
                subdirectory=RAW_BENCHMARK_DIRECTORY,
                suffix="json",
                body=body,
            )
            try:
                metadata = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ContaminationReplayV2Error(
                    "Hugging Face metadata response is not JSON"
                ) from exc
            if not isinstance(metadata, dict):
                raise ContaminationReplayV2Error("Hugging Face metadata is not an object")
            metadata_cache[spec.dataset] = (
                metadata,
                {
                    "request_url": metadata_url,
                    "http_status": status,
                    "response_date": _parse_http_date(headers.get("date")),
                    "raw_response_sha256": raw_sha,
                    "raw_response_bytes": len(body),
                    "raw_file": raw_path,
                },
            )
        metadata, metadata_receipt = metadata_cache[spec.dataset]
        if metadata.get("sha") != spec.revision:
            raise ContaminationReplayV2Error(f"{spec.dataset} revision drifted")
        if _expected_hf_license(spec) not in _hf_metadata_license(metadata):
            raise ContaminationReplayV2Error(f"{spec.dataset} license metadata drifted")

        page_receipts: list[dict[str, Any]] = []
        observed_total: int | None = None
        for offset in range(0, spec.expected_rows, 100):
            query = urllib.parse.urlencode(
                {
                    "dataset": spec.dataset,
                    "config": spec.config,
                    "split": spec.split,
                    "offset": offset,
                    "length": min(100, spec.expected_rows - offset),
                }
            )
            request_url = f"{HF_ROWS_API}?{query}"
            body, headers, status = _request_bytes(request_url)
            raw_path, raw_sha = _capture_raw(
                output_directory,
                subdirectory=RAW_BENCHMARK_DIRECTORY,
                suffix="json",
                body=body,
            )
            try:
                page = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ContaminationReplayV2Error("Hugging Face rows response is not JSON") from exc
            if not isinstance(page, dict) or not isinstance(page.get("rows"), list):
                raise ContaminationReplayV2Error("Hugging Face rows payload is malformed")
            revision_header = headers.get("x-revision")
            if revision_header != spec.revision:
                raise ContaminationReplayV2Error(f"{spec.dataset} rows revision header drifted")
            page_total = int(page.get("num_rows_total", -1))
            observed_total = page_total if observed_total is None else observed_total
            if page_total != observed_total or page_total != spec.expected_rows:
                raise ContaminationReplayV2Error(f"{spec.dataset}/{spec.config} row count drifted")
            captured_at = _parse_http_date(headers.get("date"))
            row_indexes: list[int] = []
            for result in page["rows"]:
                if not isinstance(result, Mapping) or not isinstance(result.get("row"), Mapping):
                    raise ContaminationReplayV2Error("Hugging Face row is malformed")
                row_index = int(result.get("row_idx", -1))
                row_payload = dict(result["row"])
                question = row_payload.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ContaminationReplayV2Error("benchmark row has no question text")
                row_indexes.append(row_index)
                source_id = (
                    f"hf:{spec.dataset}@{spec.revision}:{spec.config}:{spec.split}:{row_index}"
                )
                source_url = (
                    f"https://huggingface.co/datasets/{spec.dataset}/viewer/"
                    f"{urllib.parse.quote(spec.config, safe='')}/{spec.split}?row={row_index}"
                )
                records.append(
                    {
                        "source_reference_sha256": _sha256_bytes(source_id.encode("utf-8")),
                        "source_id": source_id,
                        "dataset": spec.dataset,
                        "dataset_revision": spec.revision,
                        "config": spec.config,
                        "split": spec.split,
                        "row_index": row_index,
                        "source_url": source_url,
                        "repository_revision_url": (
                            f"https://huggingface.co/datasets/{spec.dataset}/tree/{spec.revision}"
                        ),
                        "license_id": spec.license_id,
                        "license_url": spec.license_url,
                        "text": question,
                        "text_sha256": _sha256_bytes(question.encode("utf-8")),
                        "captured_at": captured_at,
                        "raw_row_payload_sha256": canonical_sha256(row_payload),
                    }
                )
            expected_indexes = list(range(offset, min(offset + 100, spec.expected_rows)))
            if row_indexes != expected_indexes:
                raise ContaminationReplayV2Error("Hugging Face rows are missing or reordered")
            page_receipts.append(
                {
                    "request_url": request_url,
                    "http_status": status,
                    "response_date": captured_at,
                    "dataset_revision_header": revision_header,
                    "offset": offset,
                    "row_count": len(row_indexes),
                    "row_index_min": row_indexes[0],
                    "row_index_max": row_indexes[-1],
                    "raw_response_sha256": raw_sha,
                    "raw_response_bytes": len(body),
                    "raw_file": raw_path,
                }
            )
        dataset_receipts.append(
            {
                "dataset": spec.dataset,
                "revision": spec.revision,
                "config": spec.config,
                "split": spec.split,
                "row_count": spec.expected_rows,
                "license_id": spec.license_id,
                "license_url": spec.license_url,
                "scope_reason": spec.scope_reason,
                "metadata_receipt": metadata_receipt,
                "page_receipts": page_receipts,
            }
        )

    exclusion_receipts: list[dict[str, Any]] = []
    for excluded in EXCLUDED_CULINARY_DATASETS:
        metadata_url = HF_DATASET_API.format(dataset=excluded["dataset"])
        body, headers, status = _request_bytes(metadata_url)
        raw_path, raw_sha = _capture_raw(
            output_directory,
            subdirectory=RAW_BENCHMARK_DIRECTORY,
            suffix="json",
            body=body,
        )
        metadata = json.loads(body)
        if not isinstance(metadata, dict) or metadata.get("sha") != excluded["expected_revision"]:
            raise ContaminationReplayV2Error("excluded culinary dataset revision drifted")
        exclusion_receipts.append(
            {
                **excluded,
                "observed_revision": str(metadata["sha"]),
                "gated": metadata.get("gated"),
                "declared_licenses": sorted(_hf_metadata_license(metadata)),
                "request_url": metadata_url,
                "http_status": status,
                "response_date": _parse_http_date(headers.get("date")),
                "raw_response_sha256": raw_sha,
                "raw_response_bytes": len(body),
                "raw_file": raw_path,
                "content_downloaded": False,
                "scan_performed": False,
            }
        )

    records.sort(key=lambda row: str(row["source_id"]))
    dataset_receipts.sort(key=lambda row: (str(row["dataset"]), str(row["config"])))
    exclusion_receipts.sort(key=lambda row: str(row["dataset"]))
    payload = {
        "schema_version": BENCHMARK_SNAPSHOT_SCHEMA,
        "artifact_role": "captured_licensed_benchmark_question_corpus",
        "campaign_sha256": PINNED_INPUTS["campaign"]["semantic_sha256"],
        "source_provider": "Hugging Face Hub and Dataset Viewer API",
        "provider_documentation_url": "https://huggingface.co/docs/dataset-viewer/rows",
        "capture_user_agent": HTTP_USER_AGENT,
        "dataset_receipts": dataset_receipts,
        "records": records,
        "excluded_culinary_dataset_receipts": exclusion_receipts,
        "coverage": {
            "captured_dataset_slices": len(dataset_receipts),
            "captured_records": len(records),
            "known_relevant_culinary_datasets_excluded": len(exclusion_receipts),
            "external_benchmark_universe_exhaustive": False,
            "all_captured_records_have_declared_license": all(
                bool(row["license_id"]) for row in records
            ),
        },
        "claim_boundary": {
            "complete_benchmark_corpus_coverage": False,
            "contamination_free": False,
            "rank_eligible": False,
        },
    }
    sealed = _seal_document(payload)
    return _write_json_artifact(sealed, output_directory, "benchmark-snapshot")


def _clean_rss_text(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", html.unescape(HTML_TAG_RE.sub(" ", value))).strip()


def _canonical_web_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((scheme, netloc, path, parsed.query, ""))


def _quoted_title_query(title: str) -> str:
    collapsed = WHITESPACE_RE.sub(" ", title).strip().replace('"', "")
    if not collapsed:
        raise ContaminationReplayV2Error("assigned source title is empty")
    return f'"{collapsed}"'


def capture_web_snapshot(
    root: Path,
    output_directory: Path,
    *,
    request_interval_seconds: float = 0.4,
) -> Path:
    """Capture one exact-title external search receipt for all 180 prompts."""

    rows = _load_assignment_rows(root)
    output_directory.mkdir(parents=True, exist_ok=True)
    query_receipts: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    collection_failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        candidate_id = str(row["candidate_id"])
        prompt = str(row["prompt"])
        source = row["source_metadata_visible_after_blind_decision"]
        title = str(source.get("title_text_deidentified", ""))
        source_url = str(source.get("url", ""))
        query_text = _quoted_title_query(title)
        query_sha256 = normalized_prompt_sha256(prompt)
        query_url = (
            BING_RSS_ENDPOINT
            + "?"
            + urllib.parse.urlencode(
                {
                    "q": query_text,
                    "format": "rss",
                    "count": "10",
                    "setlang": "en-US",
                    "cc": "us",
                    "safeSearch": "Strict",
                }
            )
        )
        body: bytes | None = None
        headers: dict[str, str] = {}
        status: int | None = None
        error_type: str | None = None
        for attempt in range(1, 3):
            try:
                body, headers, status = _request_bytes(query_url)
                break
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                error_type = type(exc).__name__
                if attempt == 1:
                    time.sleep(1.0)
        if body is None or status is None:
            collection_failures.append(
                {
                    "candidate_id": candidate_id,
                    "query_sha256": query_sha256,
                    "error_type": error_type or "unknown_network_error",
                }
            )
            continue
        raw_sha = _sha256_bytes(body)
        captured_at = _parse_http_date(headers.get("date"))
        try:
            xml_root = ET.fromstring(body)
            items = list(xml_root.findall(".//item"))
        except ET.ParseError:
            collection_failures.append(
                {
                    "candidate_id": candidate_id,
                    "query_sha256": query_sha256,
                    "error_type": "invalid_rss_xml",
                    "raw_response_sha256": raw_sha,
                }
            )
            items = []
        references: list[str] = []
        returned_urls: list[str] = []
        for rank, item in enumerate(items[:10], start=1):
            result_url = _clean_rss_text(item.findtext("link"))
            if not result_url or not urllib.parse.urlsplit(result_url).scheme.startswith("http"):
                continue
            source_reference = _sha256_bytes(f"{query_sha256}\0{rank}\0{result_url}".encode())
            references.append(source_reference)
            returned_urls.append(result_url)
            result_records.append(
                {
                    "source_reference_sha256": source_reference,
                    "candidate_id": candidate_id,
                    "query_sha256": query_sha256,
                    "result_rank": rank,
                    "source_url": result_url,
                    "source_host": (urllib.parse.urlsplit(result_url).hostname or "").casefold(),
                    "captured_at": captured_at,
                    "license_id": None,
                    "license_status": "destination_page_license_not_evaluated",
                }
            )
        canonical_source = _canonical_web_url(source_url)
        source_url_returned = any(
            _canonical_web_url(result_url) == canonical_source for result_url in returned_urls
        )
        query_receipts.append(
            {
                "candidate_id": candidate_id,
                "prompt_sha256": str(row["prompt_sha256"]),
                "query_sha256": query_sha256,
                "query_text_sha256": _sha256_bytes(query_text.encode("utf-8")),
                "query_url": query_url,
                "source_url": source_url,
                "provider_contract_sha256": WEB_PROVIDER_CONTRACT_SHA256,
                "http_status": status,
                "response_date": captured_at,
                "raw_response_sha256": raw_sha,
                "raw_response_bytes": len(body),
                "raw_response_retained": False,
                "raw_file": None,
                "result_count": len(references),
                "result_record_set_sha256": canonical_sha256(sorted(references)),
                "source_url_returned": source_url_returned,
            }
        )
        if index + 1 < len(rows) and request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    query_receipts.sort(key=lambda row: str(row["candidate_id"]))
    result_records.sort(key=lambda row: (str(row["candidate_id"]), int(row["result_rank"])))
    collection_failures.sort(key=lambda row: str(row["candidate_id"]))
    source_return_count = sum(bool(row["source_url_returned"]) for row in query_receipts)
    payload = {
        "schema_version": WEB_SNAPSHOT_SCHEMA,
        "artifact_role": "captured_external_web_exact_title_search",
        "campaign_sha256": PINNED_INPUTS["campaign"]["semantic_sha256"],
        "provider_contract": WEB_PROVIDER_CONTRACT,
        "provider_contract_sha256": WEB_PROVIDER_CONTRACT_SHA256,
        "capture_user_agent": HTTP_USER_AGENT,
        "query_receipts": query_receipts,
        "result_records": result_records,
        "collection_failures": collection_failures,
        "coverage": {
            "assigned_prompts": ASSIGNED_PROMPT_COUNT,
            "queries_attempted": ASSIGNED_PROMPT_COUNT,
            "successful_response_receipts": len(query_receipts),
            "queries_with_at_least_one_result": sum(
                int(row["result_count"]) > 0 for row in query_receipts
            ),
            "result_records": len(result_records),
            "known_positive_source_urls_returned": source_return_count,
            "known_positive_retrieval_rate_million": (
                source_return_count * 1_000_000 // ASSIGNED_PROMPT_COUNT
            ),
            "full_query_receipt_coverage": len(query_receipts) == ASSIGNED_PROMPT_COUNT,
            "provider_known_positive_validation_passed": (
                source_return_count == ASSIGNED_PROMPT_COUNT
            ),
        },
        "claim_boundary": {
            "live_search_reproducible": False,
            "captured_response_replayable": False,
            "captured_response_hashes_replayable": True,
            "result_text_redistribution_rights_confirmed": False,
            "external_web_method_suitable_for_admission": False,
            "contamination_free": False,
            "rank_eligible": False,
        },
    }
    sealed = _seal_document(payload)
    return _write_json_artifact(sealed, output_directory, "web-snapshot")


def redact_web_snapshot_for_distribution(source_snapshot: Path, output_directory: Path) -> Path:
    """Drop unlicensed result text/raw paths while preserving response commitments.

    This is a one-way publication transform for captures produced before the
    rights boundary was enforced in the collector.  The source and raw response
    files must be destroyed after this content-addressed derivative verifies.
    """

    source = _load_json_object(source_snapshot, expected_schema=WEB_SNAPSHOT_SCHEMA)
    receipts = source.get("query_receipts")
    records = source.get("result_records")
    if not isinstance(receipts, list) or not isinstance(records, list):
        raise ContaminationReplayV2Error("source web capture is malformed")
    redacted_records = [
        {
            "source_reference_sha256": str(record["source_reference_sha256"]),
            "candidate_id": str(record["candidate_id"]),
            "query_sha256": str(record["query_sha256"]),
            "result_rank": int(record["result_rank"]),
            "source_url": str(record["source_url"]),
            "source_host": str(record["source_host"]),
            "captured_at": str(record["captured_at"]),
            "license_id": None,
            "license_status": "destination_page_license_not_evaluated",
        }
        for record in records
    ]
    redacted_receipts = [
        {
            **{
                key: value
                for key, value in receipt.items()
                if key not in {"raw_file", "raw_response_retained"}
            },
            "raw_response_retained": False,
            "raw_file": None,
        }
        for receipt in receipts
    ]
    payload = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "artifact_sha256",
            "query_receipts",
            "result_records",
            "claim_boundary",
        }
    }
    payload["artifact_role"] = "captured_external_web_search_commitments_redacted"
    payload["query_receipts"] = redacted_receipts
    payload["result_records"] = redacted_records
    payload["claim_boundary"] = {
        "live_search_reproducible": False,
        "captured_response_replayable": False,
        "captured_response_hashes_replayable": True,
        "result_text_redistribution_rights_confirmed": False,
        "external_web_method_suitable_for_admission": False,
        "contamination_free": False,
        "rank_eligible": False,
    }
    return _write_json_artifact(_seal_document(payload), output_directory, "web-snapshot")


def _load_json_object(path: Path, *, expected_schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContaminationReplayV2Error("snapshot is unavailable or symlinked")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContaminationReplayV2Error("snapshot is not valid JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") != expected_schema:
        raise ContaminationReplayV2Error("snapshot schema version differs")
    if document.get("artifact_sha256") != _artifact_digest(document):
        raise ContaminationReplayV2Error("snapshot semantic digest mismatch")
    return document


def _verify_raw_receipt(snapshot_path: Path, receipt: Mapping[str, Any]) -> None:
    raw_path = _safe_raw_path(snapshot_path.parent, str(receipt.get("raw_file", "")))
    digest = _physical_sha256(raw_path)
    if digest != receipt.get("raw_response_sha256"):
        raise ContaminationReplayV2Error("raw capture digest mismatch")
    if raw_path.stat().st_size != int(receipt.get("raw_response_bytes", -1)):
        raise ContaminationReplayV2Error("raw capture byte count mismatch")


def verify_benchmark_snapshot(path: Path) -> dict[str, Any]:
    document = _load_json_object(path, expected_schema=BENCHMARK_SNAPSHOT_SCHEMA)
    receipts = document.get("dataset_receipts")
    records = document.get("records")
    exclusions = document.get("excluded_culinary_dataset_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(BENCHMARK_DATASETS):
        raise ContaminationReplayV2Error("benchmark dataset receipt coverage differs")
    if not isinstance(records, list) or len(records) != sum(
        spec.expected_rows for spec in BENCHMARK_DATASETS
    ):
        raise ContaminationReplayV2Error("benchmark record coverage differs")
    expected_slices = {
        (spec.dataset, spec.revision, spec.config, spec.split, spec.expected_rows)
        for spec in BENCHMARK_DATASETS
    }
    observed_slices: set[tuple[str, str, str, str, int]] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ContaminationReplayV2Error("benchmark receipt is malformed")
        observed_slices.add(
            (
                str(receipt.get("dataset")),
                str(receipt.get("revision")),
                str(receipt.get("config")),
                str(receipt.get("split")),
                int(receipt.get("row_count", -1)),
            )
        )
        metadata = receipt.get("metadata_receipt")
        pages = receipt.get("page_receipts")
        if not isinstance(metadata, Mapping) or not isinstance(pages, list) or not pages:
            raise ContaminationReplayV2Error("benchmark receipt has no captured pages")
        _verify_raw_receipt(path, metadata)
        for page in pages:
            if not isinstance(page, Mapping):
                raise ContaminationReplayV2Error("benchmark page receipt is malformed")
            _verify_raw_receipt(path, page)
    if observed_slices != expected_slices:
        raise ContaminationReplayV2Error("benchmark dataset slice identity differs")
    references: set[str] = set()
    source_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ContaminationReplayV2Error("benchmark record is malformed")
        reference = str(record.get("source_reference_sha256", ""))
        source_id = str(record.get("source_id", ""))
        text = record.get("text")
        if (
            not SHA256_RE.fullmatch(reference)
            or reference in references
            or not source_id
            or source_id in source_ids
            or not isinstance(text, str)
            or _sha256_bytes(text.encode("utf-8")) != record.get("text_sha256")
        ):
            raise ContaminationReplayV2Error("benchmark record integrity failed")
        references.add(reference)
        source_ids.add(source_id)
    if not isinstance(exclusions, list) or len(exclusions) != len(EXCLUDED_CULINARY_DATASETS):
        raise ContaminationReplayV2Error("culinary exclusion receipt coverage differs")
    for receipt in exclusions:
        if not isinstance(receipt, Mapping):
            raise ContaminationReplayV2Error("culinary exclusion receipt is malformed")
        _verify_raw_receipt(path, receipt)
        if (
            receipt.get("content_downloaded") is not False
            or receipt.get("scan_performed") is not False
        ):
            raise ContaminationReplayV2Error("excluded culinary dataset is misrepresented")
    boundary = document.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not False
        for field in ("complete_benchmark_corpus_coverage", "contamination_free", "rank_eligible")
    ):
        raise ContaminationReplayV2Error("benchmark snapshot overstates coverage")
    return document


def verify_web_snapshot(path: Path, assignment_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    document = _load_json_object(path, expected_schema=WEB_SNAPSHOT_SCHEMA)
    if document.get("provider_contract_sha256") != canonical_sha256(
        document.get("provider_contract")
    ):
        raise ContaminationReplayV2Error("web provider contract digest mismatch")
    receipts = document.get("query_receipts")
    records = document.get("result_records")
    failures = document.get("collection_failures")
    if (
        not isinstance(receipts, list)
        or not isinstance(records, list)
        or not isinstance(failures, list)
    ):
        raise ContaminationReplayV2Error("web snapshot collections are malformed")
    expected = {str(row["candidate_id"]): row for row in assignment_rows}
    observed: set[str] = set()
    references: set[str] = set()
    records_by_candidate: dict[str, list[str]] = {candidate_id: [] for candidate_id in expected}
    for record in records:
        if not isinstance(record, Mapping):
            raise ContaminationReplayV2Error("web result record is malformed")
        candidate_id = str(record.get("candidate_id", ""))
        reference = str(record.get("source_reference_sha256", ""))
        if (
            candidate_id not in expected
            or not SHA256_RE.fullmatch(reference)
            or reference in references
            or record.get("license_id") is not None
            or "text" in record
            or "title" in record
            or "description" in record
            or "text_sha256" in record
        ):
            raise ContaminationReplayV2Error("web result record integrity failed")
        references.add(reference)
        records_by_candidate[candidate_id].append(reference)
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ContaminationReplayV2Error("web query receipt is malformed")
        candidate_id = str(receipt.get("candidate_id", ""))
        row = expected.get(candidate_id)
        if row is None or candidate_id in observed:
            raise ContaminationReplayV2Error("web query receipt identity differs")
        observed.add(candidate_id)
        if (
            receipt.get("prompt_sha256") != row.get("prompt_sha256")
            or receipt.get("query_sha256") != normalized_prompt_sha256(str(row["prompt"]))
            or receipt.get("provider_contract_sha256") != WEB_PROVIDER_CONTRACT_SHA256
            or int(receipt.get("http_status", -1)) != 200
            or receipt.get("result_record_set_sha256")
            != canonical_sha256(sorted(records_by_candidate[candidate_id]))
            or receipt.get("raw_response_retained") is not False
            or receipt.get("raw_file") is not None
            or not SHA256_RE.fullmatch(str(receipt.get("raw_response_sha256", "")))
            or int(receipt.get("raw_response_bytes", 0)) <= 0
        ):
            raise ContaminationReplayV2Error("web query receipt does not bind captured results")
    failure_ids = {str(row.get("candidate_id")) for row in failures if isinstance(row, Mapping)}
    if observed & failure_ids or observed | failure_ids != set(expected):
        raise ContaminationReplayV2Error("web queries do not partition the 180 assignments")
    boundary = document.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not False
        for field in (
            "live_search_reproducible",
            "result_text_redistribution_rights_confirmed",
            "external_web_method_suitable_for_admission",
            "contamination_free",
            "rank_eligible",
        )
    ):
        raise ContaminationReplayV2Error("web snapshot overstates method suitability")
    return document


_SCAN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SCAN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "using",
        "with",
    }
)
_SEMANTIC_DIMENSIONS = 64


def _normalize_scan_text(value: str) -> str:
    normalized = __import__("unicodedata").normalize("NFKC", value).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    return WHITESPACE_RE.sub(" ", normalized).strip()


def _scan_tokens(value: str) -> list[str]:
    return [
        token
        for token in _SCAN_TOKEN_RE.findall(_normalize_scan_text(value))
        if len(token) > 1 and token not in _SCAN_STOPWORDS
    ][:512]


def _character_fourgrams(value: str) -> frozenset[str]:
    normalized = _normalize_scan_text(value)
    if not normalized:
        return frozenset()
    width = min(4, len(normalized))
    return frozenset(
        normalized[index : index + width] for index in range(max(1, len(normalized) - width + 1))
    )


def _word_trigrams(tokens: Sequence[str]) -> frozenset[tuple[str, ...]]:
    if not tokens:
        return frozenset()
    width = min(3, len(tokens))
    return frozenset(
        tuple(tokens[index : index + width]) for index in range(max(1, len(tokens) - width + 1))
    )


def _random_index_vector(token: str) -> np.ndarray:
    seed = hashlib.sha256(f"flavourbench-ri-v1\0{token}".encode()).digest()
    bits = np.unpackbits(np.frombuffer(seed * 2, dtype=np.uint8))[:_SEMANTIC_DIMENSIONS]
    vector = np.where(bits == 0, -1.0, 1.0).astype(np.float64)
    return vector / math.sqrt(_SEMANTIC_DIMENSIONS)


def _normalized_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _build_benchmark_scan_index(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    records = list(benchmark["records"])
    normalized_rows = [_normalize_scan_text(str(record["text"])) for record in records]
    token_rows = [_scan_tokens(str(record["text"])) for record in records]
    character_rows = [_character_fourgrams(str(record["text"])) for record in records]
    trigram_rows = [_word_trigrams(tokens) for tokens in token_rows]
    character_index: dict[str, list[int]] = defaultdict(list)
    trigram_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for record_index, grams in enumerate(character_rows):
        for gram in grams:
            character_index[gram].append(record_index)
    for record_index, trigrams in enumerate(trigram_rows):
        for trigram in trigrams:
            trigram_index[trigram].append(record_index)

    document_frequency: Counter[str] = Counter()
    for tokens in token_rows:
        document_frequency.update(set(tokens))
    document_count = max(1, len(token_rows))
    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }
    word_vectors: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
    )
    for tokens in token_rows:
        for token_index, token in enumerate(tokens):
            for context_index in range(max(0, token_index - 2), min(len(tokens), token_index + 3)):
                if context_index != token_index:
                    word_vectors[token] += _random_index_vector(tokens[context_index])
    resolved_word_vectors = {
        token: _normalized_vector(vector) for token, vector in word_vectors.items()
    }

    def embed(tokens: Sequence[str]) -> np.ndarray:
        vector = np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
        total_weight = 0.0
        for token in tokens:
            weight = idf.get(token, math.log(1 + document_count) + 1.0)
            vector += weight * resolved_word_vectors.get(token, _random_index_vector(token))
            total_weight += weight
        return _normalized_vector(vector / total_weight) if total_weight else vector

    semantic_matrix = np.stack([embed(tokens) for tokens in token_rows])
    return {
        "records": records,
        "normalized_rows": normalized_rows,
        "character_rows": character_rows,
        "trigram_rows": trigram_rows,
        "character_index": character_index,
        "trigram_index": trigram_index,
        "semantic_matrix": semantic_matrix,
        "word_vectors": resolved_word_vectors,
        "idf": idf,
        "document_count": document_count,
    }


def _query_semantic_vector(prompt: str, index: Mapping[str, Any]) -> np.ndarray:
    tokens = _scan_tokens(prompt)
    vector = np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
    total_weight = 0.0
    idf = index["idf"]
    word_vectors = index["word_vectors"]
    document_count = int(index["document_count"])
    for token in tokens:
        weight = idf.get(token, math.log(1 + document_count) + 1.0)
        vector += weight * word_vectors.get(token, _random_index_vector(token))
        total_weight += weight
    return _normalized_vector(vector / total_weight) if total_weight else vector


def _benchmark_hit(*, method: str, score: float, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "source_reference_sha256": str(record["source_reference_sha256"]),
        "matched_text_sha256": str(record["text_sha256"]),
        "similarity_milli": min(1000, max(0, int(round(score * 1000)))),
        "source_class": "benchmark_corpus",
        "source_url": str(record["source_url"]),
        "dataset": str(record["dataset"]),
        "license_id": str(record["license_id"]),
        "automated_interpretation": "human_review_trigger_not_leak_ground_truth",
    }


def _scan_benchmark_prompt(
    prompt: str,
    *,
    index: Mapping[str, Any],
    completed_at: str,
    corpus_snapshot_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = index["records"]
    normalized_prompt = _normalize_scan_text(prompt)
    prompt_characters = _character_fourgrams(prompt)
    prompt_trigrams = _word_trigrams(_scan_tokens(prompt))
    hits_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in ("exact", "fuzzy", "ngram", "semantic")
    }

    for record_index, normalized_record in enumerate(index["normalized_rows"]):
        exact_score = float(
            normalized_prompt == normalized_record
            or (len(normalized_record) >= 40 and normalized_record in normalized_prompt)
            or (len(normalized_prompt) >= 40 and normalized_prompt in normalized_record)
        )
        if exact_score >= CONTAMINATION_REPORT_THRESHOLDS["exact"]:
            hits_by_method["exact"].append(
                _benchmark_hit(method="exact", score=exact_score, record=records[record_index])
            )

    fuzzy_intersections: Counter[int] = Counter()
    for gram in prompt_characters:
        fuzzy_intersections.update(index["character_index"].get(gram, ()))
    for record_index, intersection in fuzzy_intersections.items():
        denominator = len(prompt_characters) + len(index["character_rows"][record_index])
        score = (2 * intersection / denominator) if denominator else 0.0
        if score >= CONTAMINATION_REPORT_THRESHOLDS["fuzzy"]:
            hits_by_method["fuzzy"].append(
                _benchmark_hit(method="fuzzy", score=score, record=records[record_index])
            )

    ngram_intersections: Counter[int] = Counter()
    for trigram in prompt_trigrams:
        ngram_intersections.update(index["trigram_index"].get(trigram, ()))
    for record_index, intersection in ngram_intersections.items():
        denominator = len(prompt_trigrams | index["trigram_rows"][record_index])
        score = intersection / denominator if denominator else 0.0
        if score >= CONTAMINATION_REPORT_THRESHOLDS["ngram"]:
            hits_by_method["ngram"].append(
                _benchmark_hit(method="ngram", score=score, record=records[record_index])
            )

    query_vector = _query_semantic_vector(prompt, index)
    semantic_scores = np.maximum(0.0, index["semantic_matrix"] @ query_vector)
    for record_index in np.flatnonzero(
        semantic_scores >= CONTAMINATION_REPORT_THRESHOLDS["semantic"]
    ):
        hits_by_method["semantic"].append(
            _benchmark_hit(
                method="semantic",
                score=float(semantic_scores[record_index]),
                record=records[int(record_index)],
            )
        )

    all_hits = [
        hit
        for method in ("exact", "fuzzy", "ngram", "semantic")
        for hit in sorted(
            hits_by_method[method],
            key=lambda row: (
                -int(row["similarity_milli"]),
                str(row["source_reference_sha256"]),
            ),
        )
    ]
    methods = []
    query_sha256 = normalized_prompt_sha256(prompt)
    retained_hits: list[dict[str, Any]] = []
    for method in ("exact", "fuzzy", "ngram", "semantic"):
        method_hits = [hit for hit in all_hits if hit["method"] == method]
        auto_reject_milli = round(1000 * CONTAMINATION_AUTOREJECT_THRESHOLDS[method])
        method_retained = [
            hit for hit in method_hits if int(hit["similarity_milli"]) >= auto_reject_milli
        ]
        seen_references = {str(hit["source_reference_sha256"]) for hit in method_retained}
        for hit in method_hits[:5]:
            reference = str(hit["source_reference_sha256"])
            if reference not in seen_references:
                method_retained.append(hit)
                seen_references.add(reference)
        method_retained.sort(
            key=lambda row: (
                str(row["method"]),
                -int(row["similarity_milli"]),
                str(row["source_reference_sha256"]),
            )
        )
        retained_hits.extend(method_retained)
        methods.append(
            {
                "method": method,
                "performed": True,
                "implementation_version": BENCHMARK_SCAN_IMPLEMENTATION_VERSION,
                "implementation_sha256": BENCHMARK_SCAN_IMPLEMENTATION_SHA256,
                "query_sha256": query_sha256,
                "corpus_snapshot_sha256": corpus_snapshot_sha256,
                "result_set_sha256": canonical_sha256(method_hits),
                "hit_count": len(method_hits),
                "retained_hit_count": len(method_retained),
                "retention_policy": (
                    "all auto-reject-threshold hits plus the five highest report-threshold "
                    "hits; the full ordered result set is committed by result_set_sha256"
                ),
                "completed_at": completed_at,
            }
        )
    methods.append(
        {
            "method": "web",
            "performed": False,
            "implementation_version": None,
            "implementation_sha256": None,
            "query_sha256": query_sha256,
            "corpus_snapshot_sha256": None,
            "result_set_sha256": None,
            "hit_count": 0,
            "retained_hit_count": 0,
            "retention_policy": "no result text retained; no web similarity scan performed",
            "completed_at": None,
            "reason": (
                "third-party result text and raw responses were not retained because "
                "redistribution rights were not established"
            ),
        }
    )
    return methods, retained_hits


def build_replay_v2(paths: ReplayV2Paths) -> dict[str, Any]:
    """Build the deterministic captured-method successor and issue NO-GO."""

    v1_rebuild = build_replay_artifact(paths.v1_inputs)
    if (
        v1_rebuild.get("artifact_sha256") != PINNED_REPLAY_SEMANTIC_SHA256
        or _sha256_bytes(canonical_json_bytes(v1_rebuild) + b"\n") != PINNED_REPLAY_PHYSICAL_SHA256
    ):
        raise ContaminationReplayV2Error("v1 replay no longer rebuilds to its published pins")
    assignment_rows = _load_assignment_rows(paths.root)
    benchmark = verify_benchmark_snapshot(paths.benchmark_snapshot)
    web = verify_web_snapshot(paths.web_snapshot, assignment_rows)

    full_web_receipts = len(web["query_receipts"]) == ASSIGNED_PROMPT_COUNT
    scan_records: list[dict[str, Any]] = []
    hit_counts: Counter[str] = Counter()
    hit_candidate_ids: set[str] = set()
    scan_failure_records: list[dict[str, Any]] = [
        {
            "scope": "all_assigned_prompts",
            "method": "web",
            "reason": (
                "captured response hashes and result URLs exist, but third-party result text "
                "and response bodies were not retained without redistribution rights"
            ),
        }
    ]
    scan_index = _build_benchmark_scan_index(benchmark)
    completed_at = max(str(record["captured_at"]) for record in benchmark["records"])
    benchmark_record_set_sha256 = canonical_sha256(
        [str(record["source_reference_sha256"]) for record in benchmark["records"]]
    )
    for row in assignment_rows:
        candidate_id = str(row["candidate_id"])
        methods, hits = _scan_benchmark_prompt(
            str(row["prompt"]),
            index=scan_index,
            completed_at=completed_at,
            corpus_snapshot_sha256=benchmark_record_set_sha256,
        )
        for method_receipt in methods:
            hit_counts[str(method_receipt["method"])] += int(method_receipt["hit_count"])
        if any(int(method_receipt["hit_count"]) for method_receipt in methods):
            hit_candidate_ids.add(candidate_id)
        scan_records.append(
            {
                "assignment_ordinal": int(row["assignment_ordinal"]),
                "candidate_id": candidate_id,
                "prompt_sha256": str(row["prompt_sha256"]),
                "normalized_prompt_sha256": normalized_prompt_sha256(str(row["prompt"])),
                "methods": methods,
                "hits": hits,
                "high_similarity_hit": any(
                    int(hit["similarity_milli"])
                    >= round(1000 * CONTAMINATION_AUTOREJECT_THRESHOLDS[str(hit["method"])])
                    for hit in hits
                ),
                "human_disposition": None,
            }
        )

    method_coverage = {
        method: {
            "performed": (len(scan_records) == ASSIGNED_PROMPT_COUNT and method != "web"),
            "assigned_prompts": ASSIGNED_PROMPT_COUNT,
            "prompts_scanned": len(scan_records) if method != "web" else 0,
            "coverage_percent": (
                100 if len(scan_records) == ASSIGNED_PROMPT_COUNT and method != "web" else 0
            ),
            "hit_count": int(hit_counts[method]),
        }
        for method in ("exact", "fuzzy", "ngram", "semantic", "web")
    }
    benchmark_complete = bool(benchmark["claim_boundary"]["complete_benchmark_corpus_coverage"])
    web_suitable = bool(web["claim_boundary"]["external_web_method_suitable_for_admission"])
    real_calibration_observed = False
    blocking_reasons = [
        "no independently labelled >=150-case detector calibration artifact was observed",
        (
            "known relevant culinary benchmark corpora were gated or lacked "
            "redistributable license metadata"
        ),
        "the captured benchmark corpus is a declared non-exhaustive slice of public benchmarks",
        (
            "the unauthenticated RSS search interface has no frozen service contract "
            "or confirmed result-text redistribution license"
        ),
        "the web provider did not return every known-positive attributed source URL",
    ]
    semantic_pair_count = ASSIGNED_PROMPT_COUNT * len(benchmark["records"])
    semantic_report_hit_rate_million = (
        int(hit_counts["semantic"]) * 1_000_000 // semantic_pair_count
    )
    if semantic_report_hit_rate_million >= 50_000:
        blocking_reasons.append(
            "the uncalibrated semantic detector reported "
            f"{int(hit_counts['semantic']):,} of {semantic_pair_count:,} prompt-record "
            "pairs and is not selective enough to interpret without calibration"
        )
    blocking_reasons.append(
        "web result-text replay was not performed because redistribution rights were unknown"
    )

    payload = {
        "schema_version": REPLAY_SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "status": "no_go",
        "bound_inputs": {
            "campaign": PINNED_INPUTS["campaign"],
            "review_assignment": PINNED_INPUTS["review_assignment"],
            "v1_replay": {
                "schema_version": "flavourbench-task-validation-automated-replay-v1",
                "semantic_sha256": PINNED_REPLAY_SEMANTIC_SHA256,
                "physical_sha256": PINNED_REPLAY_PHYSICAL_SHA256,
            },
            "benchmark_snapshot": {
                "schema_version": BENCHMARK_SNAPSHOT_SCHEMA,
                "semantic_sha256": str(benchmark["artifact_sha256"]),
                "physical_sha256": _physical_sha256(paths.benchmark_snapshot),
            },
            "web_snapshot": {
                "schema_version": WEB_SNAPSHOT_SCHEMA,
                "semantic_sha256": str(web["artifact_sha256"]),
                "physical_sha256": _physical_sha256(paths.web_snapshot),
            },
        },
        "policy": {
            "policy_version": POLICY_VERSION,
            "required_methods": ["exact", "fuzzy", "ngram", "semantic", "web"],
            "scan_implementation_version": BENCHMARK_SCAN_IMPLEMENTATION_VERSION,
            "scan_implementation_sha256": BENCHMARK_SCAN_IMPLEMENTATION_SHA256,
            "scan_implementation_policy": BENCHMARK_SCAN_POLICY,
            "semantic_method": "distributional-random-indexing-v1",
            "report_thresholds": CONTAMINATION_REPORT_THRESHOLDS,
            "auto_reject_thresholds": CONTAMINATION_AUTOREJECT_THRESHOLDS,
            "labeled_calibration_requirement": {
                "minimum_cases": 150,
                "minimum_cases_per_relation": 50,
                "minimum_precision": 0.95,
                "minimum_recall": 0.90,
                "minimum_paraphrase_recall": 0.85,
                "independent_human_labels_required": True,
            },
        },
        "coverage": {
            "assigned_prompts": ASSIGNED_PROMPT_COUNT,
            "scan_records": len(scan_records),
            "method_coverage": method_coverage,
            "benchmark_records": len(benchmark["records"]),
            "benchmark_dataset_slices": len(benchmark["dataset_receipts"]),
            "external_web_query_receipts": len(web["query_receipts"]),
            "external_web_result_records": len(web["result_records"]),
            "external_web_known_positive_source_urls_returned": int(
                web["coverage"]["known_positive_source_urls_returned"]
            ),
            "benchmark_record_set_sha256": benchmark_record_set_sha256,
            "scan_failure_records": scan_failure_records,
        },
        "findings": {
            "method_hit_counts": {
                method: int(hit_counts[method])
                for method in ("exact", "fuzzy", "ngram", "semantic", "web")
            },
            "automated_hit_candidate_ids": sorted(hit_candidate_ids),
            "records": scan_records,
            "detector_diagnostics": {
                "benchmark_prompt_record_pairs": semantic_pair_count,
                "semantic_report_hit_rate_million": semantic_report_hit_rate_million,
                "semantic_report_hit_rate_at_least_five_percent": (
                    semantic_report_hit_rate_million >= 50_000
                ),
                "candidates_with_auto_reject_threshold_hit": sum(
                    bool(record["high_similarity_hit"]) for record in scan_records
                ),
                "interpretation": (
                    "uncalibrated detector behavior; neither the report-threshold hits nor "
                    "their absence are contamination ground truth"
                ),
            },
            "automated_hits_are_human_ground_truth": False,
            "human_dispositions_observed": 0,
        },
        "external_evidence_assessment": {
            "benchmark_corpus_search_performed": len(scan_records) == ASSIGNED_PROMPT_COUNT,
            "benchmark_corpus_coverage_complete": benchmark_complete,
            "external_web_search_captured": full_web_receipts,
            "external_web_search_replay_performed": False,
            "external_web_provider_known_positive_validation_passed": bool(
                web["coverage"]["provider_known_positive_validation_passed"]
            ),
            "external_web_method_suitable_for_admission": web_suitable,
            "external_result_text_redistribution_rights_confirmed": False,
            "known_relevant_culinary_corpora_excluded": len(
                benchmark["excluded_culinary_dataset_receipts"]
            ),
            "model_training_membership_tested": False,
        },
        "calibration": {
            "real_labeled_calibration_artifact_observed": real_calibration_observed,
            "cases_observed": 0,
            "precision_threshold_verified": False,
            "recall_threshold_verified": False,
            "paraphrase_recall_threshold_verified": False,
            "test_fixtures_count_as_evidence": False,
        },
        "decision": {
            "full_campaign_contamination_method_requirement_satisfied": False,
            "disposition": "no_go",
            "blocking_reasons": blocking_reasons,
            "remediation": [
                (
                    "obtain license-compatible snapshots of the omitted culinary benchmark "
                    "corpora or document an independently reviewed replacement inventory"
                ),
                (
                    "replace the unauthenticated RSS capture with a contracted search export "
                    "whose result bytes may be archived and redistributed"
                ),
                (
                    "collect and independently label at least 50 exact, 50 paraphrase, and 50 "
                    "unrelated calibration cases without consulting model outputs"
                ),
                (
                    "rerun the content-addressed scan and verify precision, recall, paraphrase "
                    "recall, and known-positive web retrieval before task admission"
                ),
                (
                    "retain the independent human contamination auditor and every-hit review "
                    "required by campaign v6"
                ),
            ],
        },
        "claim_boundary": {
            "contamination_limited": True,
            "contamination_free": False,
            "official_task_bank": False,
            "rank_eligible": False,
            "task_bank_import_authorized": False,
            "campaign_audit_passed": False,
            "human_contamination_decision_observed": False,
            "model_calls": 0,
            "epicure_calls": 0,
            "paid_provider_calls": 0,
            "synthetic_tasks": 0,
        },
    }
    return _seal_document(payload)


def verify_replay_v2(document: Mapping[str, Any], paths: ReplayV2Paths) -> dict[str, Any]:
    if document.get("schema_version") != REPLAY_SCHEMA:
        raise ContaminationReplayV2Error("replay v2 schema differs")
    if document.get("artifact_sha256") != _artifact_digest(document):
        raise ContaminationReplayV2Error("replay v2 semantic digest mismatch")
    rebuilt = build_replay_v2(paths)
    if document != rebuilt:
        raise ContaminationReplayV2Error("replay v2 differs from deterministic rebuild")
    decision = document.get("decision")
    boundary = document.get("claim_boundary")
    if (
        not isinstance(decision, Mapping)
        or decision.get("disposition") != "no_go"
        or decision.get("full_campaign_contamination_method_requirement_satisfied") is not False
        or not isinstance(boundary, Mapping)
        or any(
            boundary.get(field) is not False
            for field in (
                "contamination_free",
                "official_task_bank",
                "rank_eligible",
                "task_bank_import_authorized",
                "campaign_audit_passed",
                "human_contamination_decision_observed",
            )
        )
    ):
        raise ContaminationReplayV2Error("replay v2 overstates admission authority")
    return dict(document)


def verify_pinned_replay_v2(replay_path: Path, paths: ReplayV2Paths) -> dict[str, Any]:
    """Verify all three published files, their raw benchmark inputs, and rebuild."""

    pinned_files = (
        (
            "benchmark snapshot",
            paths.benchmark_snapshot,
            PINNED_BENCHMARK_SNAPSHOT_SEMANTIC_SHA256,
            PINNED_BENCHMARK_SNAPSHOT_PHYSICAL_SHA256,
        ),
        (
            "web snapshot",
            paths.web_snapshot,
            PINNED_WEB_SNAPSHOT_SEMANTIC_SHA256,
            PINNED_WEB_SNAPSHOT_PHYSICAL_SHA256,
        ),
        (
            "replay v2",
            replay_path,
            PINNED_REPLAY_V2_SEMANTIC_SHA256,
            PINNED_REPLAY_V2_PHYSICAL_SHA256,
        ),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for role, path, expected_semantic, expected_physical in pinned_files:
        if path.is_symlink() or not path.is_file():
            raise ContaminationReplayV2Error(f"{role} is unavailable or symlinked")
        if _physical_sha256(path) != expected_physical:
            raise ContaminationReplayV2Error(f"{role} physical digest mismatch")
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("artifact_sha256") != expected_semantic
            or _artifact_digest(document) != expected_semantic
        ):
            raise ContaminationReplayV2Error(f"{role} semantic digest mismatch")
        loaded[role] = document
    verified = verify_replay_v2(loaded["replay v2"], paths)
    return {
        "artifact_sha256": str(verified["artifact_sha256"]),
        "physical_sha256": PINNED_REPLAY_V2_PHYSICAL_SHA256,
        "status": str(verified["status"]),
        "assigned_prompts": int(verified["coverage"]["assigned_prompts"]),
        "benchmark_records": int(verified["coverage"]["benchmark_records"]),
        "exact_fuzzy_ngram_semantic_coverage_percent": {
            method: int(verified["coverage"]["method_coverage"][method]["coverage_percent"])
            for method in ("exact", "fuzzy", "ngram", "semantic")
        },
        "web_replay_performed": bool(verified["coverage"]["method_coverage"]["web"]["performed"]),
        "calibration_cases_observed": int(verified["calibration"]["cases_observed"]),
        "rank_eligible": bool(verified["claim_boundary"]["rank_eligible"]),
    }


def write_replay_v2(document: dict[str, Any], output_directory: Path) -> Path:
    return _write_json_artifact(document, output_directory, "contamination-replay-v2")


def _find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ContaminationReplayV2Error(f"expected exactly one {pattern} artifact")
    return matches[0]


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_benchmark_parser = subparsers.add_parser("capture-benchmark")
    capture_benchmark_parser.add_argument("--output-directory", type=Path, required=True)
    capture_web_parser = subparsers.add_parser("capture-web")
    capture_web_parser.add_argument("--output-directory", type=Path, required=True)
    capture_web_parser.add_argument("--request-interval-seconds", type=float, default=0.4)
    redact_web_parser = subparsers.add_parser("redact-web")
    redact_web_parser.add_argument("--source-snapshot", type=Path, required=True)
    redact_web_parser.add_argument("--output-directory", type=Path, required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--capture-directory", type=Path, required=True)
    build_parser.add_argument("--output-directory", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--capture-directory", type=Path, required=True)
    verify_parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "capture-benchmark":
        print(capture_benchmark_snapshot(root, args.output_directory.resolve()))
        return 0
    if args.command == "capture-web":
        if args.request_interval_seconds < 0:
            parser.error("--request-interval-seconds must be nonnegative")
        print(
            capture_web_snapshot(
                root,
                args.output_directory.resolve(),
                request_interval_seconds=args.request_interval_seconds,
            )
        )
        return 0
    if args.command == "redact-web":
        print(
            redact_web_snapshot_for_distribution(
                args.source_snapshot.resolve(), args.output_directory.resolve()
            )
        )
        return 0
    capture_directory = args.capture_directory.resolve()
    benchmark = _find_single(capture_directory, "benchmark-snapshot-*.json")
    web = _find_single(capture_directory, "web-snapshot-*.json")
    paths = ReplayV2Paths.from_root(root, benchmark_snapshot=benchmark, web_snapshot=web)
    if args.command == "build":
        document = build_replay_v2(paths)
        print(write_replay_v2(document, args.output_directory.resolve()))
        return 0
    document = json.loads(args.replay.read_text(encoding="utf-8"))
    verify_replay_v2(document, paths)
    print(args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
