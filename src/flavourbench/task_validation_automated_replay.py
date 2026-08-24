"""Deterministic rights and prompt-risk replay for task-validation campaign v6.

The replay is deliberately local and read-only.  It verifies the six frozen
campaign inputs byte-for-byte, reconciles every scheduled question to the
1,052-record acquisition snapshot, and applies a pinned set of lexical scans to
all 180 scheduled candidates.  It does not query the source site, a model, or an
external contamination corpus.  Consequently it can verify automated evidence,
but it cannot make a human audit decision or establish that public questions are
free from model-training contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .prospective_task_acquisition import (
    ASSIGNMENT_SCHEMA,
    BUNDLE_SCHEMA,
    RECEIPT_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    verify_artifact,
)
from .task_validation_campaign import (
    CAMPAIGN_SCHEMA,
    FAMILIES,
    QUALITY_REPORT_SCHEMA,
    READINESS_SCHEMA,
)

REPLAY_SCHEMA = "flavourbench-task-validation-automated-replay-v1"
REPLAY_POLICY_VERSION = "flavourbench-task-validation-local-replay-policy-v1"
EXPECTED_LICENSE = "CC BY-SA 4.0"
EXPECTED_LICENSING_URL = "https://stackoverflow.com/help/licensing"
ZERO_CALL_FIELDS = (
    "answer_endpoint_requests",
    "model_calls",
    "epicure_calls",
    "paid_provider_calls",
    "synthetic_tasks",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GUID_RE = re.compile(r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
URL_RE = re.compile(r"https?://[^\s<>\]\[(){}]+", re.IGNORECASE)

# These physical digests bind the exact newline-terminated files, while the
# semantic digests bind their canonical JSON payloads.
PINNED_INPUTS: dict[str, dict[str, str]] = {
    "candidate_bundle": {
        "schema_version": BUNDLE_SCHEMA,
        "semantic_sha256": ("b13ab30bfb391e57a24c81d0398dc98e408d88e5a0bf21c4e758bf9271724cc3"),
        "physical_sha256": ("61ce05a666a44c04299caa321860f819ca9e58ee54131a27ca351f0bee8bfa8a"),
    },
    "review_assignment": {
        "schema_version": ASSIGNMENT_SCHEMA,
        "semantic_sha256": ("631932c0560ec417e47ff4c3ea94814ca9c944253252d3f4adcee8bd595221f9"),
        "physical_sha256": ("e4461a9bccb664e409e1c13d80ae8afc61f8b57352534d6f9c2cbd3bf32af3e3"),
    },
    "acquisition_receipt": {
        "schema_version": RECEIPT_SCHEMA,
        "semantic_sha256": ("847a95f7159ba778281fd5c20f0489a75f4655fc08a0de8075a0cba950259045"),
        "physical_sha256": ("b6a12cc39afb385d82744a6838d9f30771db2cd41d314def326440551b972c4f"),
    },
    "campaign": {
        "schema_version": CAMPAIGN_SCHEMA,
        "semantic_sha256": ("76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709"),
        "physical_sha256": ("b514203aa1924a8661a7d393ec519071c0fff85ef6dfda894fcb6065cda27c69"),
    },
    "quality_report": {
        "schema_version": QUALITY_REPORT_SCHEMA,
        "semantic_sha256": ("292492bce348a260eedf17b3cb04b041d4cc7c35aca47bc59bdf17d675a48ea8"),
        "physical_sha256": ("58b68bd879e96d072bf69fe4844b1e387d4d4e5f6c77895a948dba66bbdc39ce"),
    },
    "readiness": {
        "schema_version": READINESS_SCHEMA,
        "semantic_sha256": ("449df377dd8de515a46a80d36dffc80f1734f3e86a60a53651675fb75c9d82c0"),
        "physical_sha256": ("c64494828770dde9c9655d3531a44dea31b6454535191b1ef1ef672bc4382632"),
    },
}

# These are verifier pins, not inputs to the artifact payload, so updating them
# after materialization cannot change the replay's content address.
PINNED_REPLAY_SEMANTIC_SHA256 = "89f6dede2826e27bcd69eb764e32bd7a203b371f0098831c78c1077383383157"
PINNED_REPLAY_PHYSICAL_SHA256 = "ced66727597192342ddb978f7f48153a8fe82b0d5808d17f89c6ce42aabaaab9"

_SPECIALIST_TAGS = frozenset(
    {
        "allergy",
        "allergens",
        "cultural-difference",
        "defrosting",
        "diet",
        "food-safety",
        "health",
        "kosher",
        "medical",
        "mold",
        "nutrient-composition",
        "nutrition",
        "raw-meat",
        "religion",
        "storage-lifetime",
        "storage-method",
    }
)

_PATTERN_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "answer_leak",
        "explicit_source_answer_reference",
        re.compile(
            r"\b(?:accepted answer|answer (?:below|above)|as (?:the|an) answer "
            r"(?:says|notes)|from the answer|the answer on (?:this|that) "
            r"(?:post|question))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_resolution",
        "embedded_resolution_language",
        re.compile(
            r"(?:^|\n)\s*(?:edit|update|solved)\b[^\n]{0,180}\b(?:figured "
            r"(?:it|this) out|turns out|the (?:answer|solution) (?:is|was)|fixed "
            r"(?:it|this)|what worked)\b|\bI (?:eventually )?(?:figured "
            r"(?:it|this) out|solved (?:it|this))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_resolution",
        "edit_or_update_marker",
        re.compile(r"(?:^|\n)\s*(?:edit|update|solved)\s*[:\-]", re.IGNORECASE),
    ),
    (
        "link",
        "external_url_present",
        URL_RE,
    ),
    (
        "link",
        "linked_context_dependency",
        re.compile(
            r"\b(?:this|the|following|linked) (?:recipe|page|video|post|article|link)\b|"
            r"\b(?:recipe|instructions) (?:here|at this link)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "visual",
        "visual_context_reference",
        re.compile(
            r"\b(?:see|shown in|attached|pictured in|look at) (?:the |this |my )?"
            r"(?:image|photo|picture|screenshot|video)\b|"
            r"\b(?:image|photo|picture|screenshot|video) (?:below|above|attached)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "visual",
        "visual_medium_mentioned",
        re.compile(r"\b(?:image|photo|picture|pictured|screenshot|video)\b", re.IGNORECASE),
    ),
    (
        "specialist",
        "formal_allergen_or_intolerance",
        re.compile(
            r"\b(?:allerg(?:y|ies|ic|en|ens)|anaphyl(?:axis|actic)|celiac|coeliac|"
            r"intoleran(?:ce|t))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "specialist",
        "formal_nutrition",
        re.compile(
            r"\b(?:nutrition(?:al)?|calorie(?:s)?|kilocalorie(?:s)?|macros?|"
            r"recommended daily (?:allowance|intake)|glyc(?:a?emic|emic) index)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "specialist",
        "food_safety",
        re.compile(
            r"\b(?:safe to (?:eat|consume)|food poisoning|botulism|pathogen(?:s|ic)?|"
            r"salmonella|listeria|e\.?\s*coli|spoiled|spoilage|shelf[- ]life|"
            r"danger zone|water[- ]bath canning|pressure canning)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "specialist",
        "cultural_authenticity",
        re.compile(
            r"\b(?:authentic(?:ity|ally)?|culturally correct|real (?:italian|chinese|"
            r"indian|japanese|mexican|thai|french) way)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "specialist",
        "medical_or_legal",
        re.compile(
            r"\b(?:medical advice|doctor|physician|diagnos(?:e|is)|prescription|"
            r"legal advice|legally|lawful|regulation compliance)\b",
            re.IGNORECASE,
        ),
    ),
)


class TaskValidationReplayError(ValueError):
    """A pinned input, replay artifact, or deterministic invariant failed."""


@dataclass(frozen=True)
class ReplayInputPaths:
    candidate_bundle: Path
    review_assignment: Path
    acquisition_receipt: Path
    campaign: Path
    quality_report: Path
    readiness: Path

    @classmethod
    def from_root(cls, root: Path) -> ReplayInputPaths:
        acquisition = root / "artifacts/season1/prospective-task-acquisition-v1"
        campaign_dir = root / "artifacts/season1/task-validation-campaign-v6"
        return cls(
            candidate_bundle=acquisition
            / (
                "public-human-task-candidates-"
                + PINNED_INPUTS["candidate_bundle"]["semantic_sha256"]
                + ".json"
            ),
            review_assignment=acquisition
            / (
                "public-human-task-review-assignment-"
                + PINNED_INPUTS["review_assignment"]["semantic_sha256"]
                + ".json"
            ),
            acquisition_receipt=acquisition
            / (
                "public-api-acquisition-receipt-"
                + PINNED_INPUTS["acquisition_receipt"]["semantic_sha256"]
                + ".json"
            ),
            campaign=campaign_dir
            / ("campaign-" + PINNED_INPUTS["campaign"]["semantic_sha256"] + ".json"),
            quality_report=campaign_dir
            / ("quality-report-" + PINNED_INPUTS["quality_report"]["semantic_sha256"] + ".json"),
            readiness=campaign_dir
            / ("readiness-" + PINNED_INPUTS["readiness"]["semantic_sha256"] + ".json"),
        )

    def as_mapping(self) -> dict[str, Path]:
        return {
            "candidate_bundle": self.candidate_bundle,
            "review_assignment": self.review_assignment,
            "acquisition_receipt": self.acquisition_receipt,
            "campaign": self.campaign,
            "quality_report": self.quality_report,
            "readiness": self.readiness,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _physical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_inputs(paths: ReplayInputPaths) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for role, path in paths.as_mapping().items():
        pin = PINNED_INPUTS[role]
        if path.is_symlink() or not path.is_file():
            raise TaskValidationReplayError(f"{role} is unavailable or symlinked")
        physical = _physical_sha256(path)
        if physical != pin["physical_sha256"]:
            raise TaskValidationReplayError(f"{role} physical SHA-256 mismatch")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskValidationReplayError(f"{role} is not valid JSON") from exc
        if not isinstance(document, dict):
            raise TaskValidationReplayError(f"{role} is not a JSON object")
        try:
            verify_artifact(document, schema_version=pin["schema_version"])
        except ValueError as exc:
            raise TaskValidationReplayError(f"{role} semantic digest mismatch") from exc
        if document.get("artifact_sha256") != pin["semantic_sha256"]:
            raise TaskValidationReplayError(f"{role} is not the pinned semantic artifact")
        documents[role] = document
    return documents


def _normalized_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(TOKEN_RE.findall(normalized))


def _trigrams(value: str) -> frozenset[tuple[str, ...]]:
    words = _normalized_tokens(value)
    if not words:
        return frozenset()
    width = min(3, len(words))
    return frozenset(
        tuple(words[index : index + width]) for index in range(max(1, len(words) - width + 1))
    )


def _score_million(numerator: int, denominator: int) -> int:
    """Round a rational score to six decimals without floating-point arithmetic."""

    if denominator <= 0:
        return 1_000_000
    return (numerator * 1_000_000 + denominator // 2) // denominator


def _match_finding(*, category: str, rule_id: str, match: re.Match[str]) -> dict[str, Any]:
    matched = match.group(0)
    return {
        "category": category,
        "rule_id": rule_id,
        "evidence_sha256": _text_sha256(matched),
        "evidence_start": match.start(),
        "evidence_end": match.end(),
        "automated_interpretation": "human_review_trigger_not_ground_truth",
    }


def _validate_cross_artifact_links(documents: Mapping[str, Mapping[str, Any]]) -> None:
    bundle = documents["candidate_bundle"]
    assignment = documents["review_assignment"]
    receipt = documents["acquisition_receipt"]
    campaign = documents["campaign"]
    quality = documents["quality_report"]
    readiness = documents["readiness"]
    bundle_sha = PINNED_INPUTS["candidate_bundle"]["semantic_sha256"]
    assignment_sha = PINNED_INPUTS["review_assignment"]["semantic_sha256"]
    receipt_sha = PINNED_INPUTS["acquisition_receipt"]["semantic_sha256"]
    campaign_sha = PINNED_INPUTS["campaign"]["semantic_sha256"]
    quality_sha = PINNED_INPUTS["quality_report"]["semantic_sha256"]
    expected_links = (
        (assignment.get("source_candidate_bundle_sha256"), bundle_sha),
        (receipt.get("candidate_bundle_sha256"), bundle_sha),
        (receipt.get("assignment_artifact_sha256"), assignment_sha),
        (campaign.get("source_artifacts", {}).get("candidate_bundle_sha256"), bundle_sha),
        (campaign.get("source_artifacts", {}).get("review_assignment_sha256"), assignment_sha),
        (campaign.get("source_artifacts", {}).get("acquisition_receipt_sha256"), receipt_sha),
        (quality.get("source_artifacts", {}).get("candidate_bundle_sha256"), bundle_sha),
        (quality.get("source_artifacts", {}).get("review_assignment_sha256"), assignment_sha),
        (quality.get("source_artifacts", {}).get("acquisition_receipt_sha256"), receipt_sha),
        (readiness.get("bound_artifacts", {}).get("campaign_sha256"), campaign_sha),
        (readiness.get("bound_artifacts", {}).get("quality_report_sha256"), quality_sha),
    )
    if any(observed != expected for observed, expected in expected_links):
        raise TaskValidationReplayError("frozen artifact cross-link mismatch")
    expected_physical = {
        "candidate_bundle": PINNED_INPUTS["candidate_bundle"]["physical_sha256"],
        "review_assignment": PINNED_INPUTS["review_assignment"]["physical_sha256"],
        "acquisition_receipt": PINNED_INPUTS["acquisition_receipt"]["physical_sha256"],
    }
    for document in (campaign, quality):
        if document.get("source_artifacts", {}).get("physical_file_sha256") != expected_physical:
            raise TaskValidationReplayError("campaign input physical-hash ledger mismatch")
    if bundle.get("counts", {}).get("candidate_records") != 1052:
        raise TaskValidationReplayError("candidate bundle count drifted")
    if assignment.get("counts", {}).get("selected_for_blind_human_review") != 180:
        raise TaskValidationReplayError("review assignment count drifted")
    if campaign.get("observations") != {
        "human_ballots": 0,
        "adjudications": 0,
        "batch_audits": 0,
        "model_calls": 0,
        "epicure_calls": 0,
        "synthetic_tasks": 0,
    }:
        raise TaskValidationReplayError("campaign observations are not the frozen zero state")
    campaign_boundary = campaign.get("claim_boundary", {})
    if any(
        campaign_boundary.get(field) is not False
        for field in ("official_task_bank", "rank_eligible", "contamination_free")
    ):
        raise TaskValidationReplayError("campaign claim boundary drifted")
    quality_assessment = quality.get("assessment", {})
    if (
        quality_assessment.get("task_validity") != "not_yet_established"
        or quality_assessment.get("official_task_bank") is not False
        or quality_assessment.get("rank_eligible") is not False
    ):
        raise TaskValidationReplayError("quality-report claim boundary drifted")
    readiness_boundary = readiness.get("claim_boundary", {})
    if (
        readiness_boundary.get("human_ballots") != 0
        or readiness_boundary.get("official") is not False
        or readiness_boundary.get("rank_eligible") is not False
        or readiness_boundary.get("human_release_authority_exercised") is not False
    ):
        raise TaskValidationReplayError("readiness claim boundary drifted")


def _verify_receipt(receipt: Mapping[str, Any], source_question_ids: set[int]) -> dict[str, Any]:
    if receipt.get("acquisition_mode") != "real_public_api":
        raise TaskValidationReplayError("acquisition receipt mode drifted")
    if receipt.get("provider") != "Stack Exchange API 2.3" or receipt.get("site") != "cooking":
        raise TaskValidationReplayError("acquisition source identity drifted")
    for field in ZERO_CALL_FIELDS:
        if int(receipt.get(field, -1)) != 0:
            raise TaskValidationReplayError(f"acquisition receipt {field} is nonzero")
    requests = receipt.get("requests")
    if not isinstance(requests, list) or len(requests) != int(receipt.get("request_count", -1)):
        raise TaskValidationReplayError("acquisition request ledger is incomplete")
    revision_question_ids: list[int] = []
    question_requests = 0
    revision_requests = 0
    for ordinal, request in enumerate(requests, start=1):
        if not isinstance(request, Mapping) or int(request.get("ordinal", -1)) != ordinal:
            raise TaskValidationReplayError("acquisition request ordinals are not contiguous")
        if request.get("method") != "GET" or request.get("status_code") != 200:
            raise TaskValidationReplayError("acquisition request was not a successful GET")
        if request.get("authentication") != "public_unauthenticated":
            raise TaskValidationReplayError("acquisition request authentication drifted")
        response_sha = str(request.get("response_body_sha256", ""))
        if not SHA256_RE.fullmatch(response_sha) or int(request.get("response_bytes", 0)) <= 0:
            raise TaskValidationReplayError("acquisition response commitment is invalid")
        path = str(request.get("path", ""))
        if path == "/questions":
            question_requests += 1
        elif path.startswith("/posts/") and path.endswith("/revisions"):
            revision_requests += 1
            id_text = path.removeprefix("/posts/").removesuffix("/revisions")
            try:
                revision_question_ids.extend(int(value) for value in id_text.split(";") if value)
            except ValueError as exc:
                raise TaskValidationReplayError(
                    "revision request contains a nonnumeric post ID"
                ) from exc
        else:
            raise TaskValidationReplayError("acquisition request used an unexpected endpoint")
    if question_requests != int(receipt.get("question_endpoint_requests", -1)):
        raise TaskValidationReplayError("question endpoint receipt count mismatch")
    if revision_requests != int(receipt.get("revision_endpoint_requests", -1)):
        raise TaskValidationReplayError("revision endpoint receipt count mismatch")
    if len(revision_question_ids) != len(set(revision_question_ids)):
        raise TaskValidationReplayError("a source question appears in multiple revision requests")
    if set(revision_question_ids) != source_question_ids:
        raise TaskValidationReplayError(
            "revision requests do not cover the 1,052-question snapshot"
        )
    return {
        "real_public_api": True,
        "request_records_verified": len(requests),
        "question_request_records_verified": question_requests,
        "revision_request_records_verified": revision_requests,
        "revision_question_ids_verified": len(revision_question_ids),
        "answer_endpoint_requests_verified_zero": True,
        "model_calls_verified_zero": True,
        "epicure_calls_verified_zero": True,
        "raw_response_bodies_available_for_replay": False,
    }


def _verify_candidate_record(candidate: Mapping[str, Any]) -> None:
    embedded = candidate.get("candidate_record_sha256")
    payload = {key: value for key, value in candidate.items() if key != "candidate_record_sha256"}
    if not isinstance(embedded, str) or embedded != canonical_sha256(payload):
        raise TaskValidationReplayError("candidate record semantic digest mismatch")
    prompt = candidate.get("prompt")
    source = candidate.get("source")
    transformations = candidate.get("transformations")
    if not isinstance(prompt, str) or not isinstance(source, Mapping):
        raise TaskValidationReplayError("candidate prompt or source record is invalid")
    if (
        candidate.get("source_origin") != "exact_public_human_question"
        or candidate.get("synthetic") is not False
        or candidate.get("model_authored") is not False
        or candidate.get("model_rewritten") is not False
    ):
        raise TaskValidationReplayError("candidate authorship or source role drifted")
    candidate_boundary = candidate.get("claim_boundary", {})
    if (
        candidate_boundary.get("screen_is_human_task_validation") is not False
        or candidate_boundary.get("contamination_free_claimed") is not False
        or candidate_boundary.get("rank_eligible") is not False
        or candidate_boundary.get("public_source_may_have_entered_model_training") is not True
    ):
        raise TaskValidationReplayError("candidate claim boundary drifted")
    if not isinstance(transformations, Mapping):
        raise TaskValidationReplayError("candidate transformation ledger is invalid")
    if transformations.get("html_to_text") != "deterministic-html-to-text-v1":
        raise TaskValidationReplayError("candidate normalization version drifted")
    if transformations.get("deidentification") != "narrow-direct-contact-redaction-v1":
        raise TaskValidationReplayError("candidate deidentification version drifted")
    if transformations.get("unlogged_rewriting_performed") is not False:
        raise TaskValidationReplayError("candidate reports unlogged rewriting")
    if transformations.get("log") != []:
        raise TaskValidationReplayError(
            "v1 replay cannot reconstruct a nonempty redaction chain from the frozen payload"
        )
    title = source.get("title_text_deidentified")
    body = source.get("body_text_deidentified")
    if not isinstance(title, str) or not isinstance(body, str):
        raise TaskValidationReplayError("candidate source text is absent")
    if source.get("title_text_source_sha256") != _text_sha256(title):
        raise TaskValidationReplayError("candidate title text hash mismatch")
    if source.get("body_text_source_sha256") != _text_sha256(body):
        raise TaskValidationReplayError("candidate body text hash mismatch")
    if prompt != f"{title}\n\n{body}" or candidate.get("prompt_sha256") != _text_sha256(prompt):
        raise TaskValidationReplayError("candidate prompt reconstruction mismatch")
    for field in ("title_html_source_sha256", "body_html_source_sha256"):
        if not SHA256_RE.fullmatch(str(source.get(field, ""))):
            raise TaskValidationReplayError("candidate source HTML commitment is invalid")


def _rights_record(
    *, candidate: Mapping[str, Any], assignment: Mapping[str, Any], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    _verify_candidate_record(candidate)
    candidate_id = str(candidate["candidate_id"])
    prompt = str(candidate["prompt"])
    source = candidate["source"]
    assignment_source = assignment.get("source_metadata_visible_after_blind_decision")
    if (
        assignment.get("candidate_id") != candidate_id
        or schedule.get("candidate_id") != candidate_id
    ):
        raise TaskValidationReplayError("scheduled candidate identity mismatch")
    if assignment.get("prompt") != prompt or assignment.get("prompt_sha256") != _text_sha256(
        prompt
    ):
        raise TaskValidationReplayError("assignment prompt differs from candidate bundle")
    if assignment_source != source:
        raise TaskValidationReplayError("assignment source record differs from candidate bundle")
    if (
        assignment.get("phase") != "blind_validity"
        or assignment.get("model_outputs_visible") is not False
        or assignment.get("source_answer_text_visible") is not False
        or assignment.get("rank_eligible") is not False
        or schedule.get("rank_eligible") is not False
    ):
        raise TaskValidationReplayError("assignment blinding or eligibility boundary drifted")
    source_sha = canonical_sha256(source)
    if schedule.get("source_record_sha256") != source_sha:
        raise TaskValidationReplayError("campaign source record digest mismatch")
    question_id = source.get("question_id")
    revision_guid = str(source.get("revision_guid", ""))
    if schedule.get("source_question_id") != question_id:
        raise TaskValidationReplayError("campaign source question identity mismatch")
    if schedule.get("source_revision_guid") != revision_guid:
        raise TaskValidationReplayError("campaign source revision identity mismatch")
    if not isinstance(question_id, int) or question_id <= 0 or not GUID_RE.fullmatch(revision_guid):
        raise TaskValidationReplayError("source question or revision identifier is invalid")
    if not isinstance(source.get("revision_number"), int) or source["revision_number"] < 1:
        raise TaskValidationReplayError("source revision number is invalid")
    if not ISO_UTC_RE.fullmatch(str(source.get("revision_created_utc", ""))):
        raise TaskValidationReplayError("source revision timestamp is invalid")
    if source.get("content_license") != EXPECTED_LICENSE:
        raise TaskValidationReplayError("effective source license is not CC BY-SA 4.0")
    if source.get("revision_content_license") != EXPECTED_LICENSE:
        raise TaskValidationReplayError("terminal revision license is not CC BY-SA 4.0")
    question_license = source.get("question_content_license")
    if question_license not in {EXPECTED_LICENSE, None}:
        raise TaskValidationReplayError("question API license is unsupported")
    if source.get("licensing_url") != EXPECTED_LICENSING_URL:
        raise TaskValidationReplayError("source licensing URL drifted")
    if (
        source.get("corpus") != "Seasoned Advice (Stack Exchange)"
        or source.get("site") != "cooking"
    ):
        raise TaskValidationReplayError("source corpus identity drifted")
    if source.get("source_answer_ids_stored") is not False:
        raise TaskValidationReplayError("source answer identifiers were stored")
    if source.get("source_answer_payload_requested") is not False:
        raise TaskValidationReplayError("source answer payload was requested")
    attribution = source.get("attribution")
    if not isinstance(attribution, Mapping):
        raise TaskValidationReplayError("source attribution is absent")
    display_name = attribution.get("display_name")
    profile_url = attribution.get("profile_url")
    source_user_id = attribution.get("source_user_id")
    if not isinstance(display_name, str) or not display_name.strip():
        raise TaskValidationReplayError("source attribution display name is absent")
    if not isinstance(profile_url, str) or not profile_url.startswith(
        "https://cooking.stackexchange.com/users/"
    ):
        raise TaskValidationReplayError("source attribution profile URL is invalid")
    if not isinstance(source_user_id, int) or source_user_id <= 0:
        raise TaskValidationReplayError("source attribution user ID is invalid")
    source_url = str(source.get("url", ""))
    parsed = urlparse(source_url)
    expected_prefix = f"/questions/{question_id}/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "cooking.stackexchange.com"
        or not parsed.path.startswith(expected_prefix)
    ):
        raise TaskValidationReplayError("source question URL does not bind the question ID")
    checks = {
        "candidate_record_hash": True,
        "prompt_reconstruction_and_hash": True,
        "assignment_exact_match": True,
        "campaign_schedule_exact_match": True,
        "terminal_revision_identity": True,
        "effective_and_revision_license": True,
        "attribution_complete": True,
        "source_url_identity": True,
        "source_answer_payload_absent": True,
    }
    anomalies = [] if question_license == EXPECTED_LICENSE else ["question_api_license_missing"]
    return {
        "schedule_ordinal": int(schedule["schedule_ordinal"]),
        "candidate_id": candidate_id,
        "scheduling_family": str(schedule["scheduling_family"]),
        "candidate_record_sha256": str(candidate["candidate_record_sha256"]),
        "prompt_sha256": str(candidate["prompt_sha256"]),
        "source_record_sha256": source_sha,
        "source_question_id": question_id,
        "source_revision_guid": revision_guid,
        "source_revision_number": int(source["revision_number"]),
        "source_revision_created_utc": str(source["revision_created_utc"]),
        "effective_license": EXPECTED_LICENSE,
        "question_api_license": question_license,
        "terminal_revision_license": EXPECTED_LICENSE,
        "attribution_sha256": canonical_sha256(attribution),
        "source_url_sha256": _text_sha256(source_url),
        "title_text_source_sha256": str(source["title_text_source_sha256"]),
        "body_text_source_sha256": str(source["body_text_source_sha256"]),
        "title_html_source_sha256": str(source["title_html_source_sha256"]),
        "body_html_source_sha256": str(source["body_html_source_sha256"]),
        "checks": checks,
        "anomalies": anomalies,
        "automated_interpretation": "integrity_evidence_not_human_rights_decision",
    }


def _duplicate_findings(
    *,
    candidate: Mapping[str, Any],
    corpus_features: Sequence[Mapping[str, Any]],
    selected: set[str],
) -> list[dict[str, Any]]:
    candidate_id = str(candidate["candidate_id"])
    tokens = _normalized_tokens(str(candidate["prompt"]))
    trigrams = _trigrams(str(candidate["prompt"]))
    findings: list[dict[str, Any]] = []
    for other in corpus_features:
        other_id = str(other["candidate_id"])
        if other_id == candidate_id:
            continue
        scope = "scheduled_slate" if other_id in selected else "captured_source_snapshot"
        if tokens == other["tokens"]:
            findings.append(
                {
                    "category": "duplicate",
                    "rule_id": "normalized_exact_duplicate",
                    "matched_candidate_id": other_id,
                    "matched_prompt_sha256": other["prompt_sha256"],
                    "match_scope": scope,
                    "similarity_million": 1_000_000,
                    "automated_interpretation": "human_review_trigger_not_ground_truth",
                }
            )
            continue
        other_trigrams = other["trigrams"]
        if len(trigrams) < 4 or len(other_trigrams) < 4:
            continue
        intersection = len(trigrams & other_trigrams)
        union = len(trigrams | other_trigrams)
        jaccard = _score_million(intersection, union)
        containment = _score_million(intersection, min(len(trigrams), len(other_trigrams)))
        if jaccard > 820_000:
            findings.append(
                {
                    "category": "duplicate",
                    "rule_id": "trigram_jaccard_above_0_82",
                    "matched_candidate_id": other_id,
                    "matched_prompt_sha256": other["prompt_sha256"],
                    "match_scope": scope,
                    "similarity_million": jaccard,
                    "automated_interpretation": "human_review_trigger_not_ground_truth",
                }
            )
        elif min(len(trigrams), len(other_trigrams)) >= 12 and containment >= 800_000:
            findings.append(
                {
                    "category": "duplicate",
                    "rule_id": "trigram_containment_at_least_0_80",
                    "matched_candidate_id": other_id,
                    "matched_prompt_sha256": other["prompt_sha256"],
                    "match_scope": scope,
                    "similarity_million": containment,
                    "automated_interpretation": "human_review_trigger_not_ground_truth",
                }
            )
    return findings


def _scan_record(
    *,
    candidate: Mapping[str, Any],
    corpus_features: Sequence[Mapping[str, Any]],
    selected: set[str],
) -> dict[str, Any]:
    prompt = str(candidate["prompt"])
    source = candidate["source"]
    findings = _duplicate_findings(
        candidate=candidate, corpus_features=corpus_features, selected=selected
    )
    for category, rule_id, pattern in _PATTERN_RULES:
        match = pattern.search(prompt)
        if match is not None:
            findings.append(_match_finding(category=category, rule_id=rule_id, match=match))
    specialist_tags = sorted(
        {str(tag).casefold() for tag in source.get("tags", [])} & _SPECIALIST_TAGS
    )
    for tag in specialist_tags:
        findings.append(
            {
                "category": "specialist",
                "rule_id": "specialist_source_tag",
                "evidence_sha256": _text_sha256(tag),
                "automated_interpretation": "human_review_trigger_not_ground_truth",
            }
        )
    findings.sort(
        key=lambda item: (
            str(item["category"]),
            str(item["rule_id"]),
            str(item.get("matched_candidate_id", "")),
            str(item.get("evidence_sha256", "")),
        )
    )
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "prompt_sha256": str(candidate["prompt_sha256"]),
        "public_source_exposure_baseline": True,
        "findings": findings,
        "automated_disposition": "requires_human_review" if findings else "no_rule_hit",
        "human_decision": None,
    }


def _audit_sample(campaign: Mapping[str, Any], audit_kind: str) -> dict[str, Any]:
    campaign_sha = str(campaign["artifact_sha256"])
    selected: list[str] = []
    for family in FAMILIES:
        rows = [row for row in campaign["candidate_schedule"] if row["scheduling_family"] == family]
        ordered = sorted(
            rows,
            key=lambda row: _text_sha256(
                f"{campaign_sha}\0{audit_kind}\0seeded-sample-v1\0{row['candidate_id']}"
            ),
        )
        selected.extend(str(row["candidate_id"]) for row in ordered[:6])
    return {
        "sample_algorithm": "sha256-order-within-scheduling-family-v1",
        "sample_seed_commitment_sha256": _text_sha256(
            f"{campaign_sha}\0{audit_kind}\0seeded-sample-v1"
        ),
        "sample_candidate_ids": selected,
    }


def _policy_document() -> dict[str, Any]:
    return {
        "policy_version": REPLAY_POLICY_VERSION,
        "text_normalization": "Unicode NFKC, casefold, ASCII alphanumeric tokenization",
        "duplicate_methods": [
            {
                "rule_id": "normalized_exact_duplicate",
                "scope": "each scheduled prompt against the other 1,051 captured prompts",
            },
            {
                "rule_id": "trigram_jaccard_above_0_82",
                "threshold_million": 820_000,
                "comparison": "strictly_greater_than",
            },
            {
                "rule_id": "trigram_containment_at_least_0_80",
                "threshold_million": 800_000,
                "minimum_smaller_trigram_count": 12,
                "comparison": "greater_than_or_equal",
            },
        ],
        "pattern_rules": [
            {
                "category": category,
                "rule_id": rule_id,
                "pattern": pattern.pattern,
                "flags": ["IGNORECASE"] if pattern.flags & re.IGNORECASE else [],
            }
            for category, rule_id, pattern in _PATTERN_RULES
        ],
        "specialist_tags": sorted(_SPECIALIST_TAGS),
        "score_arithmetic": "integer_millionths_half_up",
        "coverage_denominator": 180,
        "automated_findings_are_human_review_triggers": True,
        "automated_findings_are_human_ground_truth": False,
    }


def build_replay_artifact(paths: ReplayInputPaths) -> dict[str, Any]:
    """Rebuild the complete replay artifact from six exact frozen inputs."""

    documents = _load_pinned_inputs(paths)
    _validate_cross_artifact_links(documents)
    bundle = documents["candidate_bundle"]
    assignment = documents["review_assignment"]
    receipt = documents["acquisition_receipt"]
    campaign = documents["campaign"]
    candidates = bundle.get("candidates")
    assignment_rows = assignment.get("assignment_rows")
    schedule = campaign.get("candidate_schedule")
    if not isinstance(candidates, list) or len(candidates) != 1052:
        raise TaskValidationReplayError("candidate snapshot is not exactly 1,052 records")
    if not isinstance(assignment_rows, list) or len(assignment_rows) != 180:
        raise TaskValidationReplayError("assignment is not exactly 180 records")
    if not isinstance(schedule, list) or len(schedule) != 180:
        raise TaskValidationReplayError("campaign schedule is not exactly 180 records")
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    corpus_features: list[dict[str, Any]] = []
    source_question_ids: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TaskValidationReplayError("candidate snapshot contains a non-object")
        _verify_candidate_record(candidate)
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in candidate_by_id:
            raise TaskValidationReplayError("candidate snapshot contains a duplicate identity")
        source = candidate["source"]
        question_id = source.get("question_id")
        if not isinstance(question_id, int) or question_id in source_question_ids:
            raise TaskValidationReplayError("candidate snapshot contains a duplicate question")
        candidate_by_id[candidate_id] = candidate
        source_question_ids.add(question_id)
        corpus_features.append(
            {
                "candidate_id": candidate_id,
                "prompt_sha256": candidate["prompt_sha256"],
                "tokens": _normalized_tokens(str(candidate["prompt"])),
                "trigrams": _trigrams(str(candidate["prompt"])),
            }
        )
    assignment_ids: set[str] = set()
    rights_records: list[dict[str, Any]] = []
    scan_records: list[dict[str, Any]] = []
    selected_ids = {str(row.get("candidate_id", "")) for row in assignment_rows}
    if len(selected_ids) != 180:
        raise TaskValidationReplayError("assignment candidate identities are not unique")
    family_counts: Counter[str] = Counter()
    for ordinal, (assignment_row, schedule_row) in enumerate(
        zip(assignment_rows, schedule, strict=True), start=1
    ):
        if not isinstance(assignment_row, Mapping) or not isinstance(schedule_row, Mapping):
            raise TaskValidationReplayError("assignment or schedule contains a non-object")
        candidate_id = str(assignment_row.get("candidate_id", ""))
        if candidate_id in assignment_ids or candidate_id not in candidate_by_id:
            raise TaskValidationReplayError("assignment candidate identity is invalid")
        if int(assignment_row.get("assignment_ordinal", -1)) != ordinal:
            raise TaskValidationReplayError("assignment ordinals are not contiguous")
        if int(schedule_row.get("schedule_ordinal", -1)) != ordinal:
            raise TaskValidationReplayError("campaign schedule ordinals are not contiguous")
        family = str(schedule_row.get("scheduling_family", ""))
        if assignment_row.get("allocation_family_hidden_from_blind_reviewer") != family:
            raise TaskValidationReplayError("assignment and schedule family differ")
        if family not in FAMILIES:
            raise TaskValidationReplayError("unknown scheduling family")
        family_counts[family] += 1
        candidate = candidate_by_id[candidate_id]
        rights_records.append(
            _rights_record(candidate=candidate, assignment=assignment_row, schedule=schedule_row)
        )
        scan_records.append(
            _scan_record(
                candidate=candidate,
                corpus_features=corpus_features,
                selected=selected_ids,
            )
        )
        assignment_ids.add(candidate_id)
    if family_counts != Counter({family: 45 for family in FAMILIES}):
        raise TaskValidationReplayError("campaign schedule is not balanced 45 per family")
    receipt_verification = _verify_receipt(receipt, source_question_ids)
    rights_anomaly_ids = sorted(
        record["candidate_id"] for record in rights_records if record["anomalies"]
    )
    scan_hit_ids = sorted(record["candidate_id"] for record in scan_records if record["findings"])
    findings_by_category: Counter[str] = Counter()
    findings_by_rule: Counter[str] = Counter()
    for record in scan_records:
        for finding in record["findings"]:
            findings_by_category[str(finding["category"])] += 1
            findings_by_rule[str(finding["rule_id"])] += 1
    rights_sample = _audit_sample(campaign, "rights")
    contamination_sample = _audit_sample(campaign, "contamination")
    rights_required = sorted(set(rights_sample["sample_candidate_ids"]) | set(rights_anomaly_ids))
    contamination_required = sorted(
        set(contamination_sample["sample_candidate_ids"]) | set(scan_hit_ids)
    )
    artifact: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA,
        "artifact_role": "automated_evidence_for_independent_human_batch_audits",
        "bound_inputs": {role: dict(pin) for role, pin in sorted(PINNED_INPUTS.items())},
        "policy": _policy_document(),
        "coverage": {
            "captured_source_records_verified": len(candidates),
            "scheduled_records_verified": len(rights_records),
            "scheduled_records_scanned": len(scan_records),
            "scheduled_records_by_family": dict(sorted(family_counts.items())),
            "rights_coverage_percent": 100,
            "scan_coverage_percent": 100,
            "scan_categories": [
                "public_source_exposure_baseline",
                "duplicate",
                "answer_leak",
                "self_resolution",
                "link",
                "visual",
                "specialist",
            ],
        },
        "acquisition_receipt_verification": receipt_verification,
        "rights": {
            "automatedEvidenceVerified": True,
            "records": rights_records,
            "integrity_failure_count": 0,
            "anomaly_candidate_ids": rights_anomaly_ids,
            "human_decision": None,
        },
        "contamination_and_prompt_risk": {
            "automatedEvidenceVerified": True,
            "records": scan_records,
            "finding_count_by_category": dict(sorted(findings_by_category.items())),
            "finding_count_by_rule": dict(sorted(findings_by_rule.items())),
            "automated_hit_candidate_ids": scan_hit_ids,
            "public_source_candidate_count": len(scan_records),
            "contamination_free": False,
            "human_decision": None,
        },
        "human_audit_handoff": {
            "rights": {
                **rights_sample,
                "anomaly_or_hit_candidate_ids": rights_anomaly_ids,
                "required_candidate_ids": rights_required,
            },
            "contamination": {
                **contamination_sample,
                "anomaly_or_hit_candidate_ids": scan_hit_ids,
                "required_candidate_ids": contamination_required,
            },
            "human_auditors_must_inspect_seeded_sample_and_every_anomaly_or_hit": True,
            "replay_sets_human_decision": False,
        },
        "runtime_projection": {
            "rights_automatedEvidenceVerified": True,
            "contamination_automatedEvidenceVerified": True,
            "contamination_automated_hit_candidate_ids": scan_hit_ids,
            "human_decision_fields_must_remain_unchanged": True,
            "campaign_audit_passed": False,
            "task_bank_import_authorized": False,
            "rank_eligible": False,
        },
        "limitations": {
            "public_source_contamination_limited_not_contamination_free": True,
            "model_training_membership_tested": False,
            "external_benchmark_corpus_tested": False,
            "external_web_search_performed": False,
            "raw_api_response_bodies_preserved": False,
            "raw_source_html_reconstructable": False,
            "recorded_html_hash_commitments_verified_for_shape_only": True,
            "source_revision_independently_refetched": False,
            "task_validity_established": False,
            "human_rights_decision_observed": False,
            "human_contamination_decision_observed": False,
            "release_go_issued": False,
        },
        "claim_boundary": {
            "official_task_bank": False,
            "rank_eligible": False,
            "automated_evidence_is_human_decision": False,
            "contamination_free": False,
            "model_or_epicure_calls": 0,
            "synthetic_tasks": 0,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def verify_replay_document(document: Mapping[str, Any], paths: ReplayInputPaths) -> dict[str, Any]:
    """Purely rebuild and compare a replay document; return a bounded runtime projection."""

    if document.get("schema_version") != REPLAY_SCHEMA:
        raise TaskValidationReplayError("replay schema version mismatch")
    embedded = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if not isinstance(embedded, str) or embedded != canonical_sha256(payload):
        raise TaskValidationReplayError("replay semantic digest mismatch")
    rebuilt = build_replay_artifact(paths)
    if document != rebuilt:
        raise TaskValidationReplayError("replay document differs from deterministic rebuild")
    projection = document["runtime_projection"]
    if (
        projection.get("rights_automatedEvidenceVerified") is not True
        or projection.get("contamination_automatedEvidenceVerified") is not True
        or projection.get("human_decision_fields_must_remain_unchanged") is not True
        or projection.get("campaign_audit_passed") is not False
        or projection.get("task_bank_import_authorized") is not False
        or projection.get("rank_eligible") is not False
    ):
        raise TaskValidationReplayError("runtime projection exceeds the automated evidence role")
    return {
        "artifact_sha256": embedded,
        "rights_automatedEvidenceVerified": True,
        "contamination_automatedEvidenceVerified": True,
        "contamination_automated_hit_candidate_ids": list(
            projection["contamination_automated_hit_candidate_ids"]
        ),
        "human_decision_fields_must_remain_unchanged": True,
        "campaign_audit_passed": False,
        "task_bank_import_authorized": False,
        "rank_eligible": False,
        "contamination_free": False,
    }


def verify_pinned_replay(path: Path, inputs: ReplayInputPaths) -> dict[str, Any]:
    """Verify the published replay bytes, semantic digest, and deterministic rebuild."""

    if path.is_symlink() or not path.is_file():
        raise TaskValidationReplayError("replay artifact is unavailable or symlinked")
    physical = _physical_sha256(path)
    if PINNED_REPLAY_PHYSICAL_SHA256 == "TO_BE_MATERIALIZED":
        raise TaskValidationReplayError("replay physical digest has not been pinned")
    if physical != PINNED_REPLAY_PHYSICAL_SHA256:
        raise TaskValidationReplayError("replay physical SHA-256 mismatch")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskValidationReplayError("replay artifact is invalid JSON") from exc
    if not isinstance(document, dict):
        raise TaskValidationReplayError("replay artifact is not a JSON object")
    if document.get("artifact_sha256") != PINNED_REPLAY_SEMANTIC_SHA256:
        raise TaskValidationReplayError("replay semantic artifact is not pinned")
    result = verify_replay_document(document, inputs)
    return {**result, "physical_sha256": physical}


def write_replay_artifact(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Atomically write one canonical, content-addressed replay artifact."""

    embedded = str(document.get("artifact_sha256", ""))
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if not SHA256_RE.fullmatch(embedded) or embedded != canonical_sha256(payload):
        raise TaskValidationReplayError("cannot write an invalid replay document")
    rendered = canonical_json_bytes(document) + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"automated-replay-{embedded}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise TaskValidationReplayError("content-addressed replay path contains other bytes")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def _root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="rebuild the pinned replay")
    build.add_argument("--root", type=Path, default=_root_from_module())
    build.add_argument("--output-dir", type=Path)
    verify = subparsers.add_parser("verify", help="verify the published replay")
    verify.add_argument("--root", type=Path, default=_root_from_module())
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = ReplayInputPaths.from_root(args.root.resolve())
    if args.command == "build":
        document = build_replay_artifact(inputs)
        output_dir = args.output_dir or (
            args.root / "artifacts/season1/task-validation-campaign-v6"
        )
        destination = write_replay_artifact(document, output_dir.resolve())
        print(
            json.dumps(
                {
                    "path": str(destination),
                    "semantic_sha256": document["artifact_sha256"],
                    "physical_sha256": _physical_sha256(destination),
                },
                sort_keys=True,
            )
        )
        return 0
    result = verify_pinned_replay(args.artifact.resolve(), inputs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
