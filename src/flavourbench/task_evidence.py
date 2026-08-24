from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VALIDATOR_DSL_VERSION = "flavourbench-validator-dsl-v1"
VALIDATOR_CONTRACT_SCHEMA_VERSION = "flavourbench-task-validator-contract-v1"
CONTAMINATION_AUDIT_SCHEMA_VERSION = "flavourbench-task-contamination-audit-v2"
CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION = "flavourbench-contamination-scan-bundle-v1"
CONTAMINATION_SCAN_IMPLEMENTATION_VERSION = "flavourbench-contamination-replay-v1"
VALIDATOR_RECEIPT_SCHEMA_VERSION = "flavourbench-task-validator-receipt-v1"
CONTAMINATION_RECEIPT_SCHEMA_VERSION = "flavourbench-contamination-receipt-v1"
TASK_EVIDENCE_REVIEW_SCHEMA_VERSION = "flavourbench-task-evidence-review-v1"
TASK_EVIDENCE_ROOT_SCHEMA_VERSION = "flavourbench-task-evidence-root-v2"
TASK_EVIDENCE_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

REQUIRED_CONTAMINATION_METHODS = frozenset({"exact", "fuzzy", "ngram", "semantic", "web"})
SHA256_PATTERN = r"^[0-9a-f]{64}$"
OCI_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"

CONTAMINATION_REPORT_THRESHOLDS = {
    "exact": 1.0,
    "fuzzy": 0.76,
    "ngram": 0.42,
    "semantic": 0.68,
    "web": 0.76,
}
CONTAMINATION_AUTOREJECT_THRESHOLDS = {
    "exact": 1.0,
    "fuzzy": 0.92,
    "ngram": 0.82,
    "semantic": 0.90,
    "web": 0.92,
}


class TaskEvidenceError(ValueError):
    """A task-evidence artifact is not reproducible or not bound to its task."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()


class RequiredEntityRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["required_entity"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    aliases: list[str] = Field(min_length=1, max_length=24)
    minimum_mentions: int = Field(default=1, ge=1, le=12, alias="minimumMentions")


class ProhibitedClaimRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["prohibited_claim"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    phrases: list[str] = Field(min_length=1, max_length=32)
    negation_aware: bool = Field(default=True, alias="negationAware")


class NumericRangeRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["numeric_range"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    anchor_aliases: list[str] = Field(min_length=1, max_length=24, alias="anchorAliases")
    unit_group: Literal[
        "mass_g", "volume_ml", "temperature_c", "duration_min", "percentage", "count"
    ] = Field(alias="unitGroup")
    minimum: Decimal
    maximum: Decimal
    max_distance_chars: int = Field(default=120, ge=16, le=500, alias="maxDistanceChars")

    @model_validator(mode="after")
    def validate_interval(self) -> NumericRangeRule:
        if not self.minimum.is_finite() or not self.maximum.is_finite():
            raise ValueError("numeric bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("numeric minimum cannot exceed maximum")
        return self


class RatioRangeRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["ratio_range"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    numerator_aliases: list[str] = Field(min_length=1, max_length=16, alias="numeratorAliases")
    denominator_aliases: list[str] = Field(min_length=1, max_length=16, alias="denominatorAliases")
    minimum_ratio: Decimal = Field(alias="minimumRatio")
    maximum_ratio: Decimal = Field(alias="maximumRatio")

    @model_validator(mode="after")
    def validate_interval(self) -> RatioRangeRule:
        if (
            not self.minimum_ratio.is_finite()
            or not self.maximum_ratio.is_finite()
            or self.minimum_ratio < 0
            or self.minimum_ratio > self.maximum_ratio
        ):
            raise ValueError("ratio bounds must form a finite nonnegative interval")
        return self


class OrderedStepsRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["ordered_steps"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    steps: list[list[str]] = Field(min_length=2, max_length=16)

    @field_validator("steps")
    @classmethod
    def validate_step_aliases(cls, value: list[list[str]]) -> list[list[str]]:
        if any(not aliases or len(aliases) > 16 for aliases in value):
            raise ValueError("every ordered step needs one to sixteen aliases")
        return value


class EvidenceCalibrationRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: Literal["evidence_calibration"]
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="ruleId")
    description: str = Field(min_length=3, max_length=500)
    qualifier_phrases: list[str] = Field(min_length=1, max_length=32, alias="qualifierPhrases")
    overclaim_phrases: list[str] = Field(min_length=1, max_length=32, alias="overclaimPhrases")


TaskValidatorRule = Annotated[
    RequiredEntityRule
    | ProhibitedClaimRule
    | NumericRangeRule
    | RatioRangeRule
    | OrderedStepsRule
    | EvidenceCalibrationRule,
    Field(discriminator="kind"),
]


class ValidatorFixture(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    fixture_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$", alias="fixtureId")
    response_text: str = Field(min_length=1, max_length=20000, alias="responseText")
    expected_rule_status: dict[str, Literal["pass", "fail"]] = Field(
        min_length=1, max_length=64, alias="expectedRuleStatus"
    )


class TaskValidatorContractArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[VALIDATOR_CONTRACT_SCHEMA_VERSION] = Field(alias="schemaVersion")
    artifact_sha256: str = Field(pattern=SHA256_PATTERN, alias="artifactSha256")
    task_public_id: str = Field(min_length=3, max_length=80, alias="taskPublicId")
    task_family: Literal["substitution", "composition", "cookability", "evidence"] = Field(
        alias="taskFamily"
    )
    task_revision: int = Field(ge=1, le=100, alias="taskRevision")
    prompt_sha256: str = Field(pattern=SHA256_PATTERN, alias="promptSha256")
    objective_scope: Literal["executable_subset", "human_only"] = Field(alias="objectiveScope")
    human_only_reason: str | None = Field(
        default=None, min_length=10, max_length=1000, alias="humanOnlyReason"
    )
    validator_dsl_version: Literal[VALIDATOR_DSL_VERSION] = Field(alias="validatorDslVersion")
    evaluator_implementation_sha256: str = Field(
        pattern=SHA256_PATTERN, alias="evaluatorImplementationSha256"
    )
    validator_container_image_digest: str = Field(
        pattern=OCI_DIGEST_PATTERN, alias="validatorContainerImageDigest"
    )
    rules: list[TaskValidatorRule] = Field(default_factory=list, max_length=64)
    fixtures: list[ValidatorFixture] = Field(default_factory=list, max_length=128)
    fixture_set_sha256: str = Field(pattern=SHA256_PATTERN, alias="fixtureSetSha256")
    status: Literal["verified"]
    verified_at: datetime = Field(alias="verifiedAt")
    verifier_reviewer_id: str = Field(min_length=3, max_length=160, alias="verifierReviewerId")

    @model_validator(mode="after")
    def validate_contract_shape(self) -> TaskValidatorContractArtifact:
        rule_ids = [rule.rule_id for rule in self.rules]
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("validator rule IDs must be unique")
        if len(set(fixture_ids)) != len(fixture_ids):
            raise ValueError("validator fixture IDs must be unique")
        if self.objective_scope == "human_only":
            if self.rules or self.fixtures or not self.human_only_reason:
                raise ValueError("human-only contracts require a reason and no executable rules")
        else:
            if not self.rules or len(self.fixtures) < 2 or self.human_only_reason is not None:
                raise ValueError(
                    "executable contracts require rules, at least two fixtures, and no "
                    "human-only reason"
                )
            expected_ids = set(rule_ids)
            for fixture in self.fixtures:
                if set(fixture.expected_rule_status) != expected_ids:
                    raise ValueError("every fixture must declare an outcome for every rule")
            for rule_id in rule_ids:
                covered = {fixture.expected_rule_status[rule_id] for fixture in self.fixtures}
                if covered != {"pass", "fail"}:
                    raise ValueError(
                        f"fixtures must exercise both pass and fail for rule {rule_id}"
                    )
        expected_fixture_sha256 = canonical_sha256(
            [fixture.model_dump(mode="json", by_alias=True) for fixture in self.fixtures]
        )
        if self.fixture_set_sha256 != expected_fixture_sha256:
            raise ValueError("fixtureSetSha256 does not match the inline fixtures")
        return self


class ContaminationCorpusRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source_reference_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="sourceReferenceSha256",
    )
    source_class: Literal["benchmark_corpus", "web_snapshot"] = Field(alias="sourceClass")
    text: str = Field(min_length=1, max_length=20000)
    text_sha256: str = Field(pattern=SHA256_PATTERN, alias="textSha256")
    query_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
        alias="querySha256",
    )
    first_published_at: datetime | None = Field(default=None, alias="firstPublishedAt")
    captured_at: datetime = Field(alias="capturedAt")

    @model_validator(mode="after")
    def validate_corpus_record(self) -> ContaminationCorpusRecord:
        if self.text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("textSha256 does not match the corpus text")
        if (self.source_class == "web_snapshot") != (self.query_sha256 is not None):
            raise ValueError("web records require a querySha256; benchmark records forbid it")
        return self


class WebCollectionReceipt(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query_sha256: str = Field(pattern=SHA256_PATTERN, alias="querySha256")
    provider: str = Field(min_length=3, max_length=120)
    provider_contract_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="providerContractSha256",
    )
    raw_response_sha256: str = Field(pattern=SHA256_PATTERN, alias="rawResponseSha256")
    result_record_set_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="resultRecordSetSha256",
    )
    collected_at: datetime = Field(alias="collectedAt")


class ContaminationScanBundle(BaseModel):
    """Frozen replay corpus for task-contamination admission.

    Web retrieval happens before task admission.  Admission replays the search over the
    captured provider response so the result is deterministic and auditable.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION] = Field(alias="schemaVersion")
    artifact_sha256: str = Field(pattern=SHA256_PATTERN, alias="artifactSha256")
    created_at: datetime = Field(alias="createdAt")
    benchmark_snapshot_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="benchmarkSnapshotSha256",
    )
    web_snapshot_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="webSnapshotSha256",
    )
    benchmark_records: list[ContaminationCorpusRecord] = Field(
        min_length=1,
        max_length=50000,
        alias="benchmarkRecords",
    )
    web_records: list[ContaminationCorpusRecord] = Field(
        min_length=1,
        max_length=10000,
        alias="webRecords",
    )
    web_collection_receipts: list[WebCollectionReceipt] = Field(
        min_length=1,
        max_length=5000,
        alias="webCollectionReceipts",
    )
    semantic_method: Literal["distributional-random-indexing-v1"] = Field(alias="semanticMethod")
    report_thresholds: dict[str, float] = Field(alias="reportThresholds")
    auto_reject_thresholds: dict[str, float] = Field(alias="autoRejectThresholds")

    @model_validator(mode="after")
    def validate_scan_bundle(self) -> ContaminationScanBundle:
        if any(record.source_class != "benchmark_corpus" for record in self.benchmark_records):
            raise ValueError("benchmarkRecords contain a non-benchmark source")
        if any(record.source_class != "web_snapshot" for record in self.web_records):
            raise ValueError("webRecords contain a non-web source")
        all_records = [*self.benchmark_records, *self.web_records]
        references = [record.source_reference_sha256 for record in all_records]
        if len(references) != len(set(references)):
            raise ValueError("contamination source references must be unique")
        if sum(len(record.text.encode("utf-8")) for record in all_records) > 64_000_000:
            raise ValueError("contamination scan bundle exceeds the 64 MB text limit")
        benchmark_snapshot = canonical_sha256(
            [
                record.model_dump(mode="json", by_alias=True)
                for record in sorted(
                    self.benchmark_records,
                    key=lambda item: item.source_reference_sha256,
                )
            ]
        )
        web_snapshot = canonical_sha256(
            [
                record.model_dump(mode="json", by_alias=True)
                for record in sorted(
                    self.web_records,
                    key=lambda item: item.source_reference_sha256,
                )
            ]
        )
        if self.benchmark_snapshot_sha256 != benchmark_snapshot:
            raise ValueError("benchmarkSnapshotSha256 does not match benchmarkRecords")
        if self.web_snapshot_sha256 != web_snapshot:
            raise ValueError("webSnapshotSha256 does not match webRecords")
        records_by_query: dict[str, list[str]] = defaultdict(list)
        for record in self.web_records:
            assert record.query_sha256 is not None
            records_by_query[record.query_sha256].append(record.source_reference_sha256)
        receipts_by_query = {
            receipt.query_sha256: receipt for receipt in self.web_collection_receipts
        }
        if len(receipts_by_query) != len(self.web_collection_receipts):
            raise ValueError("web collection receipts require unique query digests")
        if set(receipts_by_query) != set(records_by_query):
            raise ValueError("every frozen web query requires exactly one collection receipt")
        for query_sha256, references_for_query in records_by_query.items():
            if receipts_by_query[query_sha256].result_record_set_sha256 != canonical_sha256(
                sorted(references_for_query)
            ):
                raise ValueError("web collection receipt does not bind its result records")
        if self.report_thresholds != CONTAMINATION_REPORT_THRESHOLDS:
            raise ValueError("reportThresholds differ from the frozen scanner policy")
        if self.auto_reject_thresholds != CONTAMINATION_AUTOREJECT_THRESHOLDS:
            raise ValueError("autoRejectThresholds differ from the frozen scanner policy")
        return self


class ContaminationMethodReceipt(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    method: Literal["exact", "fuzzy", "ngram", "semantic", "web"]
    implementation_version: str = Field(min_length=3, max_length=160, alias="implementationVersion")
    query_sha256: str = Field(pattern=SHA256_PATTERN, alias="querySha256")
    corpus_snapshot_sha256: str = Field(pattern=SHA256_PATTERN, alias="corpusSnapshotSha256")
    result_set_sha256: str = Field(pattern=SHA256_PATTERN, alias="resultSetSha256")
    hit_count: int = Field(ge=0, le=100000, alias="hitCount")
    completed_at: datetime = Field(alias="completedAt")


class ContaminationHit(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    method: Literal["exact", "fuzzy", "ngram", "semantic", "web"]
    source_reference_sha256: str = Field(pattern=SHA256_PATTERN, alias="sourceReferenceSha256")
    matched_text_sha256: str = Field(pattern=SHA256_PATTERN, alias="matchedTextSha256")
    similarity_milli: int = Field(ge=0, le=1000, alias="similarityMilli")
    disposition: Literal[
        "unrelated", "allowed_common_phrase", "confirmed_leak", "requires_revision"
    ]
    disposition_reviewer_id: str = Field(
        min_length=3, max_length=160, alias="dispositionReviewerId"
    )


class TaskContaminationAuditArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[CONTAMINATION_AUDIT_SCHEMA_VERSION] = Field(alias="schemaVersion")
    artifact_sha256: str = Field(pattern=SHA256_PATTERN, alias="artifactSha256")
    task_public_id: str = Field(min_length=3, max_length=80, alias="taskPublicId")
    task_family: Literal["substitution", "composition", "cookability", "evidence"] = Field(
        alias="taskFamily"
    )
    task_revision: int = Field(ge=1, le=100, alias="taskRevision")
    prompt_sha256: str = Field(pattern=SHA256_PATTERN, alias="promptSha256")
    normalized_prompt_sha256: str = Field(pattern=SHA256_PATTERN, alias="normalizedPromptSha256")
    audit_implementation_sha256: str = Field(
        pattern=SHA256_PATTERN, alias="auditImplementationSha256"
    )
    scan_bundle_sha256: str = Field(pattern=SHA256_PATTERN, alias="scanBundleSha256")
    audit_container_image_digest: str = Field(
        pattern=OCI_DIGEST_PATTERN, alias="auditContainerImageDigest"
    )
    methods: list[ContaminationMethodReceipt] = Field(min_length=5, max_length=5)
    hits: list[ContaminationHit] = Field(default_factory=list, max_length=500)
    conclusion: Literal["pass", "reject"]
    auditor_reviewer_id: str = Field(min_length=3, max_length=160, alias="auditorReviewerId")
    observed_at: datetime = Field(alias="observedAt")

    @model_validator(mode="after")
    def validate_audit_shape(self) -> TaskContaminationAuditArtifact:
        methods = [receipt.method for receipt in self.methods]
        if len(set(methods)) != len(methods) or set(methods) != REQUIRED_CONTAMINATION_METHODS:
            raise ValueError("contamination audit must contain each required method exactly once")
        reported_counts = {receipt.method: receipt.hit_count for receipt in self.methods}
        observed_counts = {
            method: sum(hit.method == method for hit in self.hits)
            for method in REQUIRED_CONTAMINATION_METHODS
        }
        if reported_counts != observed_counts:
            raise ValueError("contamination method hit counts do not match inline hit records")
        adverse = any(
            hit.disposition in {"confirmed_leak", "requires_revision"} for hit in self.hits
        )
        if (self.conclusion == "pass") == adverse:
            raise ValueError("contamination conclusion is inconsistent with hit dispositions")
        return self


def _artifact_payload(artifact: BaseModel) -> dict[str, object]:
    return artifact.model_dump(
        mode="json",
        by_alias=True,
        exclude={"artifact_sha256"},
    )


def artifact_sha256(artifact: BaseModel) -> str:
    return canonical_sha256(_artifact_payload(artifact))


def load_contamination_scan_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ContaminationScanBundle:
    bundle_path = Path(path)
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise TaskEvidenceError("contamination scan bundle is unavailable")
    try:
        document = json.loads(bundle_path.read_bytes())
        bundle = ContaminationScanBundle.model_validate(document)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TaskEvidenceError("contamination scan bundle is not schema-valid") from exc
    observed_sha256 = artifact_sha256(bundle)
    if (
        not re.fullmatch(SHA256_PATTERN, expected_sha256)
        or bundle.artifact_sha256 != observed_sha256
        or not hmac_compare(bundle.artifact_sha256, expected_sha256)
    ):
        raise TaskEvidenceError("contamination scan bundle digest mismatch")
    return bundle


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time comparison without importing application-level auth helpers."""

    return hmac.compare_digest(left, right)


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
_SEMANTIC_CACHE: dict[
    str,
    tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]],
] = {}


def _scan_tokens(value: str) -> list[str]:
    return [
        token
        for token in _SCAN_TOKEN_RE.findall(normalize_text(value))
        if len(token) > 1 and token not in _SCAN_STOPWORDS
    ][:512]


def _word_trigrams(tokens: list[str]) -> set[tuple[str, ...]]:
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _random_index_vector(token: str) -> np.ndarray:
    seed = hashlib.sha256(f"flavourbench-ri-v1\0{token}".encode()).digest()
    bits = np.unpackbits(np.frombuffer(seed * 2, dtype=np.uint8))[:_SEMANTIC_DIMENSIONS]
    vector = np.where(bits == 0, -1.0, 1.0).astype(np.float64)
    return vector / math.sqrt(_SEMANTIC_DIMENSIONS)


def _normalized_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _semantic_index(
    bundle: ContaminationScanBundle,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    cached = _SEMANTIC_CACHE.get(bundle.artifact_sha256)
    if cached is not None:
        return cached
    records = [*bundle.benchmark_records, *bundle.web_records]
    token_rows = {record.source_reference_sha256: _scan_tokens(record.text) for record in records}
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in token_rows.values():
        for token in set(tokens):
            document_frequency[token] += 1
    document_count = max(1, len(token_rows))
    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }
    word_vectors: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
    )
    for tokens in token_rows.values():
        for index, token in enumerate(tokens):
            for context_index in range(max(0, index - 2), min(len(tokens), index + 3)):
                if context_index != index:
                    word_vectors[token] += _random_index_vector(tokens[context_index])
    resolved_word_vectors = {
        token: _normalized_vector(vector) for token, vector in word_vectors.items()
    }

    def embed(tokens: list[str]) -> np.ndarray:
        vector = np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
        total_weight = 0.0
        for token in tokens:
            weight = idf.get(token, math.log(1 + document_count) + 1.0)
            vector += weight * resolved_word_vectors.get(token, _random_index_vector(token))
            total_weight += weight
        return _normalized_vector(vector / total_weight) if total_weight else vector

    record_vectors = {reference: embed(tokens) for reference, tokens in token_rows.items()}
    resolved = (resolved_word_vectors, idf, record_vectors)
    if len(_SEMANTIC_CACHE) >= 4:
        _SEMANTIC_CACHE.pop(next(iter(_SEMANTIC_CACHE)))
    _SEMANTIC_CACHE[bundle.artifact_sha256] = resolved
    return resolved


def _semantic_query_vector(
    prompt: str,
    bundle: ContaminationScanBundle,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    word_vectors, idf, record_vectors = _semantic_index(bundle)
    tokens = _scan_tokens(prompt)
    vector = np.zeros(_SEMANTIC_DIMENSIONS, dtype=np.float64)
    total_weight = 0.0
    document_count = len(bundle.benchmark_records) + len(bundle.web_records)
    for token in tokens:
        weight = idf.get(token, math.log(1 + max(1, document_count)) + 1.0)
        vector += weight * word_vectors.get(token, _random_index_vector(token))
        total_weight += weight
    if total_weight:
        vector = _normalized_vector(vector / total_weight)
    return vector, record_vectors


def _raw_contamination_hit(
    *,
    method: str,
    record: ContaminationCorpusRecord,
    score: float,
) -> dict[str, object]:
    return {
        "method": method,
        "source_reference_sha256": record.source_reference_sha256,
        "matched_text_sha256": record.text_sha256,
        "similarity_milli": min(1000, max(0, int(round(score * 1000)))),
    }


def replay_contamination_scan(
    prompt: str,
    bundle: ContaminationScanBundle,
    *,
    completed_at: datetime,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Execute all five frozen contamination methods over the pinned bundle."""

    if bundle.artifact_sha256 != artifact_sha256(bundle):
        raise TaskEvidenceError("contamination scan bundle digest mismatch")
    query_sha256 = normalized_prompt_sha256(prompt)
    normalized_prompt = normalize_text(prompt)
    prompt_tokens = _scan_tokens(prompt)
    prompt_trigrams = _word_trigrams(prompt_tokens)
    query_vector, record_vectors = _semantic_query_vector(prompt, bundle)
    hits_by_method: dict[str, list[dict[str, object]]] = {
        method: [] for method in REQUIRED_CONTAMINATION_METHODS
    }

    for record in bundle.benchmark_records:
        normalized_record = normalize_text(record.text)
        record_tokens = _scan_tokens(record.text)
        exact_score = float(
            normalized_prompt == normalized_record
            or (len(normalized_prompt) >= 40 and normalized_prompt in normalized_record)
        )
        fuzzy_score = SequenceMatcher(None, normalized_prompt, normalized_record).ratio()
        ngram_score = _jaccard(prompt_trigrams, _word_trigrams(record_tokens))
        semantic_score = max(
            0.0,
            float(np.dot(query_vector, record_vectors[record.source_reference_sha256])),
        )
        for method, score in (
            ("exact", exact_score),
            ("fuzzy", fuzzy_score),
            ("ngram", ngram_score),
            ("semantic", semantic_score),
        ):
            if score >= CONTAMINATION_REPORT_THRESHOLDS[method]:
                hits_by_method[method].append(
                    _raw_contamination_hit(method=method, record=record, score=score)
                )

    web_records = [record for record in bundle.web_records if record.query_sha256 == query_sha256]
    if not web_records:
        raise TaskEvidenceError("frozen web snapshot has no receipt for this prompt query")
    for record in web_records:
        normalized_record = normalize_text(record.text)
        exact_score = float(
            normalized_prompt == normalized_record
            or (len(normalized_prompt) >= 40 and normalized_prompt in normalized_record)
        )
        fuzzy_score = SequenceMatcher(None, normalized_prompt, normalized_record).ratio()
        ngram_score = _jaccard(prompt_trigrams, _word_trigrams(_scan_tokens(record.text)))
        semantic_score = max(
            0.0,
            float(np.dot(query_vector, record_vectors[record.source_reference_sha256])),
        )
        web_score = max(exact_score, fuzzy_score, ngram_score, semantic_score)
        if web_score >= CONTAMINATION_REPORT_THRESHOLDS["web"]:
            hits_by_method["web"].append(
                _raw_contamination_hit(method="web", record=record, score=web_score)
            )

    raw_hits = [
        hit
        for method in sorted(REQUIRED_CONTAMINATION_METHODS)
        for hit in sorted(
            hits_by_method[method],
            key=lambda row: (str(row["source_reference_sha256"]), -int(row["similarity_milli"])),
        )
    ]
    methods: list[dict[str, object]] = []
    for method in sorted(REQUIRED_CONTAMINATION_METHODS):
        method_hits = [hit for hit in raw_hits if hit["method"] == method]
        methods.append(
            {
                "method": method,
                "implementationVersion": CONTAMINATION_SCAN_IMPLEMENTATION_VERSION,
                "querySha256": query_sha256,
                "corpusSnapshotSha256": (
                    bundle.web_snapshot_sha256
                    if method == "web"
                    else bundle.benchmark_snapshot_sha256
                ),
                "resultSetSha256": canonical_sha256(method_hits),
                "hitCount": len(method_hits),
                "completedAt": completed_at.isoformat(),
            }
        )
    return methods, raw_hits


_NEGATION_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bavoid\b|\bomit\b|"
    r"\bexclude\b|\bdoes\s+not\b|\bdo\s+not\b|\bcannot\b|\bcan't\b|\bdoesn't\b)"
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    normalized = normalize_text(phrase)
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 48) : start]
    clause = re.split(r"[.!?;:\n]", prefix)[-1]
    return bool(_NEGATION_RE.search(clause))


def _unnegated_spans(text: str, phrases: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for phrase in phrases:
        for match in _phrase_pattern(phrase).finditer(text):
            if not _is_negated(text, match.start()):
                spans.append((match.start(), match.end(), phrase))
    return sorted(spans)


_UNICODE_FRACTIONS = {
    "¼": Decimal("0.25"),
    "½": Decimal("0.5"),
    "¾": Decimal("0.75"),
    "⅓": Decimal(1) / Decimal(3),
    "⅔": Decimal(2) / Decimal(3),
    "⅛": Decimal("0.125"),
    "⅜": Decimal("0.375"),
    "⅝": Decimal("0.625"),
    "⅞": Decimal("0.875"),
}
_NUMBER = r"(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+|[¼½¾⅓⅔⅛⅜⅝⅞])"
_QUANTITY_RE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<unit>kg|kilograms?|g|grams?|mg|milligrams?|"
    rf"l|lit(?:re|er)s?|ml|millilit(?:re|er)s?|tsp|teaspoons?|tbsp|tablespoons?|"
    rf"cups?|fl\.?\s*oz|°?c|celsius|°?f|fahrenheit|seconds?|secs?|minutes?|mins?|"
    rf"hours?|hrs?|%|percent)\b"
)


def _decimal(value: str) -> Decimal | None:
    compact = value.replace(" ", "")
    if compact in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[compact]
    if "/" in compact:
        numerator, denominator = compact.split("/", 1)
        try:
            denominator_value = Decimal(denominator)
            return Decimal(numerator) / denominator_value if denominator_value else None
        except InvalidOperation:
            return None
    try:
        return Decimal(compact)
    except InvalidOperation:
        return None


def _convert_quantity(value: Decimal, unit: str) -> tuple[str, Decimal] | None:
    unit = unit.replace(".", "").replace(" ", "")
    if unit in {"kg", "kilogram", "kilograms"}:
        return "mass_g", value * 1000
    if unit in {"g", "gram", "grams"}:
        return "mass_g", value
    if unit in {"mg", "milligram", "milligrams"}:
        return "mass_g", value / 1000
    if unit in {"l", "litre", "litres", "liter", "liters"}:
        return "volume_ml", value * 1000
    if unit in {"ml", "millilitre", "millilitres", "milliliter", "milliliters"}:
        return "volume_ml", value
    if unit in {"tsp", "teaspoon", "teaspoons"}:
        return "volume_ml", value * Decimal("4.92892159375")
    if unit in {"tbsp", "tablespoon", "tablespoons"}:
        return "volume_ml", value * Decimal("14.78676478125")
    if unit in {"cup", "cups"}:
        return "volume_ml", value * Decimal("236.5882365")
    if unit in {"floz"}:
        return "volume_ml", value * Decimal("29.5735295625")
    if unit in {"c", "°c", "celsius"}:
        return "temperature_c", value
    if unit in {"f", "°f", "fahrenheit"}:
        return "temperature_c", (value - 32) * Decimal(5) / Decimal(9)
    if unit in {"second", "seconds", "sec", "secs"}:
        return "duration_min", value / 60
    if unit in {"minute", "minutes", "min", "mins"}:
        return "duration_min", value
    if unit in {"hour", "hours", "hr", "hrs"}:
        return "duration_min", value * 60
    if unit in {"%", "percent"}:
        return "percentage", value
    return None


def _near_anchor(text: str, start: int, end: int, aliases: list[str], distance: int) -> bool:
    return any(
        match.end() >= start - distance and match.start() <= end + distance
        for alias in aliases
        for match in _phrase_pattern(alias).finditer(text)
        if not _is_negated(text, match.start())
    )


def _numeric_values(text: str, rule: NumericRangeRule) -> list[Decimal]:
    values: list[Decimal] = []
    if rule.unit_group == "count":
        number_re = re.compile(_NUMBER)
        for match in number_re.finditer(text):
            value = _decimal(match.group())
            if value is not None and _near_anchor(
                text,
                match.start(),
                match.end(),
                rule.anchor_aliases,
                rule.max_distance_chars,
            ):
                values.append(value)
        return values
    for match in _QUANTITY_RE.finditer(text):
        value = _decimal(match.group("number"))
        converted = _convert_quantity(value, match.group("unit")) if value is not None else None
        if (
            converted is not None
            and converted[0] == rule.unit_group
            and _near_anchor(
                text,
                match.start(),
                match.end(),
                rule.anchor_aliases,
                rule.max_distance_chars,
            )
        ):
            values.append(converted[1])
    return values


def _alias_expression(aliases: list[str]) -> str:
    return (
        "(?:"
        + "|".join(
            re.escape(normalize_text(alias)).replace(r"\ ", r"\s+")
            for alias in sorted(aliases, key=len, reverse=True)
        )
        + ")"
    )


def _ratio_values(text: str, rule: RatioRangeRule) -> list[Decimal]:
    numerator = _alias_expression(rule.numerator_aliases)
    denominator = _alias_expression(rule.denominator_aliases)
    connectors = r"(?:to|and|:|/)"
    patterns = (
        re.compile(
            rf"(?P<a>{_NUMBER})\s*(?:parts?\s+)?{numerator}.{{0,48}}?{connectors}"
            rf"\s*(?P<b>{_NUMBER})\s*(?:parts?\s+)?{denominator}"
        ),
        re.compile(
            rf"{numerator}\s*(?P<a>{_NUMBER})?.{{0,32}}?{connectors}\s*"
            rf"{denominator}\s*(?P<b>{_NUMBER})"
        ),
    )
    values: list[Decimal] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            left = _decimal(match.group("a") or "1")
            right = _decimal(match.group("b"))
            if left is not None and right not in {None, Decimal(0)}:
                values.append(left / right)
    return values


def evaluate_rules(
    rules: list[TaskValidatorRule],
    response_text: str,
) -> list[dict[str, object]]:
    text = normalize_text(response_text)
    results: list[dict[str, object]] = []
    for rule in rules:
        evidence: dict[str, object]
        if isinstance(rule, RequiredEntityRule):
            spans = _unnegated_spans(text, rule.aliases)
            passed = len(spans) >= rule.minimum_mentions
            evidence = {"unnegated_mentions": len(spans)}
        elif isinstance(rule, ProhibitedClaimRule):
            spans = (
                _unnegated_spans(text, rule.phrases)
                if rule.negation_aware
                else [
                    (match.start(), match.end(), phrase)
                    for phrase in rule.phrases
                    for match in _phrase_pattern(phrase).finditer(text)
                ]
            )
            passed = not spans
            evidence = {"prohibited_unnegated_mentions": len(spans)}
        elif isinstance(rule, NumericRangeRule):
            values = _numeric_values(text, rule)
            passed = any(rule.minimum <= value <= rule.maximum for value in values)
            evidence = {
                "candidate_values": [format(value, "f") for value in values],
                "canonical_unit_group": rule.unit_group,
            }
        elif isinstance(rule, RatioRangeRule):
            values = _ratio_values(text, rule)
            passed = any(rule.minimum_ratio <= value <= rule.maximum_ratio for value in values)
            evidence = {"candidate_ratios": [format(value, "f") for value in values]}
        elif isinstance(rule, OrderedStepsRule):
            positions: list[int] = []
            cursor = 0
            for aliases in rule.steps:
                candidates = [
                    match.start()
                    for alias in aliases
                    for match in _phrase_pattern(alias).finditer(text, cursor)
                    if not _is_negated(text, match.start())
                ]
                if not candidates:
                    break
                cursor = min(candidates) + 1
                positions.append(cursor - 1)
            passed = len(positions) == len(rule.steps)
            evidence = {"ordered_steps_found": len(positions), "required": len(rule.steps)}
        elif isinstance(rule, EvidenceCalibrationRule):
            qualifier_count = len(_unnegated_spans(text, rule.qualifier_phrases))
            overclaim_count = len(_unnegated_spans(text, rule.overclaim_phrases))
            passed = qualifier_count > 0 and overclaim_count == 0
            evidence = {
                "qualifier_mentions": qualifier_count,
                "unnegated_overclaims": overclaim_count,
            }
        else:  # pragma: no cover - Pydantic closes the discriminated union.
            raise TaskEvidenceError(f"unsupported validator rule: {type(rule).__name__}")
        results.append(
            {
                "rule_id": rule.rule_id,
                "kind": rule.kind,
                "status": "pass" if passed else "fail",
                "evidence": evidence,
            }
        )
    return results


def _seal_receipt(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def evaluate_contract(
    contract: TaskValidatorContractArtifact,
    response_text: str,
) -> dict[str, object]:
    results = evaluate_rules(contract.rules, response_text)
    payload: dict[str, object] = {
        "schema_version": VALIDATOR_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": contract.artifact_sha256,
        "task_public_id": contract.task_public_id,
        "task_revision": contract.task_revision,
        "prompt_sha256": contract.prompt_sha256,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "evaluator_implementation_sha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "validator_dsl_version": VALIDATOR_DSL_VERSION,
        "objective_scope": contract.objective_scope,
        "status": (
            "not_applicable"
            if contract.objective_scope == "human_only"
            else "pass"
            if all(result["status"] == "pass" for result in results)
            else "fail"
        ),
        "rule_results": results,
    }
    return _seal_receipt(payload)


def verify_validator_contract(
    contract: TaskValidatorContractArtifact,
    *,
    task_public_id: str,
    task_family: str,
    task_revision: int,
    prompt_sha256: str,
    objective_validator_possible: bool,
    expected_container_image_digest: str,
) -> dict[str, object]:
    if contract.artifact_sha256 != artifact_sha256(contract):
        raise TaskEvidenceError("validator artifact digest mismatch")
    expected_binding = (task_public_id, task_family, task_revision, prompt_sha256)
    observed_binding = (
        contract.task_public_id,
        contract.task_family,
        contract.task_revision,
        contract.prompt_sha256,
    )
    if observed_binding != expected_binding:
        raise TaskEvidenceError("validator artifact is bound to a different task revision")
    if contract.evaluator_implementation_sha256 != TASK_EVIDENCE_IMPLEMENTATION_SHA256:
        raise TaskEvidenceError("validator implementation digest does not match this evaluator")
    if (
        not re.fullmatch(OCI_DIGEST_PATTERN, expected_container_image_digest)
        or contract.validator_container_image_digest != expected_container_image_digest
    ):
        raise TaskEvidenceError("validator container digest is unresolved or mismatched")
    expected_scope = "executable_subset" if objective_validator_possible else "human_only"
    if contract.objective_scope != expected_scope:
        raise TaskEvidenceError("validator scope contradicts the sealed task candidate")
    fixture_receipts: list[dict[str, object]] = []
    for fixture in contract.fixtures:
        result_by_rule = {
            str(result["rule_id"]): str(result["status"])
            for result in evaluate_rules(contract.rules, fixture.response_text)
        }
        if result_by_rule != fixture.expected_rule_status:
            raise TaskEvidenceError(
                f"validator fixture {fixture.fixture_id} does not reproduce expected outcomes"
            )
        fixture_receipts.append(
            {
                "fixture_id": fixture.fixture_id,
                "response_sha256": hashlib.sha256(
                    fixture.response_text.encode("utf-8")
                ).hexdigest(),
                "rule_status": result_by_rule,
            }
        )
    payload: dict[str, object] = {
        "schema_version": VALIDATOR_RECEIPT_SCHEMA_VERSION,
        "receipt_class": "contract_fixture_verification",
        "contract_sha256": contract.artifact_sha256,
        "task_public_id": task_public_id,
        "task_revision": task_revision,
        "prompt_sha256": prompt_sha256,
        "evaluator_implementation_sha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "container_image_digest": expected_container_image_digest,
        "rule_count": len(contract.rules),
        "fixture_count": len(contract.fixtures),
        "fixture_set_sha256": contract.fixture_set_sha256,
        "fixture_receipts_sha256": canonical_sha256(fixture_receipts),
        "status": "verified",
    }
    return _seal_receipt(payload)


def verify_contamination_audit(
    audit: TaskContaminationAuditArtifact,
    *,
    scan_bundle: ContaminationScanBundle,
    prompt: str,
    task_public_id: str,
    task_family: str,
    task_revision: int,
    prompt_sha256: str,
    expected_container_image_digest: str,
    forbidden_reviewer_ids: set[str],
) -> dict[str, object]:
    if audit.artifact_sha256 != artifact_sha256(audit):
        raise TaskEvidenceError("contamination artifact digest mismatch")
    expected_binding = (task_public_id, task_family, task_revision, prompt_sha256)
    observed_binding = (
        audit.task_public_id,
        audit.task_family,
        audit.task_revision,
        audit.prompt_sha256,
    )
    if observed_binding != expected_binding:
        raise TaskEvidenceError("contamination artifact is bound to a different task revision")
    normalized_sha256 = normalized_prompt_sha256(prompt)
    if audit.normalized_prompt_sha256 != normalized_sha256:
        raise TaskEvidenceError("contamination audit normalized-prompt digest mismatch")
    if audit.audit_implementation_sha256 != TASK_EVIDENCE_IMPLEMENTATION_SHA256:
        raise TaskEvidenceError("contamination implementation digest does not match")
    if (
        scan_bundle.artifact_sha256 != artifact_sha256(scan_bundle)
        or audit.scan_bundle_sha256 != scan_bundle.artifact_sha256
    ):
        raise TaskEvidenceError("contamination audit is not bound to the replay corpus")
    if (
        not re.fullmatch(OCI_DIGEST_PATTERN, expected_container_image_digest)
        or audit.audit_container_image_digest != expected_container_image_digest
    ):
        raise TaskEvidenceError("contamination container digest is unresolved or mismatched")
    if audit.auditor_reviewer_id in forbidden_reviewer_ids:
        raise TaskEvidenceError("contamination auditor must be distinct from author and approvers")
    if any(hit.disposition_reviewer_id != audit.auditor_reviewer_id for hit in audit.hits):
        raise TaskEvidenceError("every contamination hit requires the sealed auditor disposition")
    if any(receipt.query_sha256 != normalized_sha256 for receipt in audit.methods):
        raise TaskEvidenceError("contamination method receipt is bound to another query")
    replay_method_payloads, replay_hits = replay_contamination_scan(
        prompt,
        scan_bundle,
        completed_at=audit.observed_at,
    )
    try:
        replay_methods = [
            ContaminationMethodReceipt.model_validate(payload) for payload in replay_method_payloads
        ]
    except ValueError as exc:  # pragma: no cover - scanner output is locally constructed.
        raise TaskEvidenceError("replayed contamination receipt is invalid") from exc
    observed_method_payloads = [
        receipt.model_dump(mode="json", by_alias=True)
        for receipt in sorted(audit.methods, key=lambda item: item.method)
    ]
    expected_method_payloads = [
        receipt.model_dump(mode="json", by_alias=True)
        for receipt in sorted(replay_methods, key=lambda item: item.method)
    ]
    if observed_method_payloads != expected_method_payloads:
        raise TaskEvidenceError("contamination method receipts do not reproduce")
    observed_hits = sorted(
        [
            {
                "method": hit.method,
                "source_reference_sha256": hit.source_reference_sha256,
                "matched_text_sha256": hit.matched_text_sha256,
                "similarity_milli": hit.similarity_milli,
            }
            for hit in audit.hits
        ],
        key=lambda row: (
            str(row["method"]),
            str(row["source_reference_sha256"]),
            -int(row["similarity_milli"]),
        ),
    )
    if observed_hits != replay_hits:
        raise TaskEvidenceError("contamination hit records do not reproduce")
    auto_reject_hits = [
        hit
        for hit in replay_hits
        if int(hit["similarity_milli"])
        >= round(CONTAMINATION_AUTOREJECT_THRESHOLDS[str(hit["method"])] * 1000)
    ]
    if auto_reject_hits:
        raise TaskEvidenceError(
            "exact or high-similarity contamination hits require task rejection"
        )
    if audit.conclusion != "pass":
        raise TaskEvidenceError("confirmatory tasks require a passing contamination audit")
    payload: dict[str, object] = {
        "schema_version": CONTAMINATION_RECEIPT_SCHEMA_VERSION,
        "audit_sha256": audit.artifact_sha256,
        "task_public_id": task_public_id,
        "task_revision": task_revision,
        "prompt_sha256": prompt_sha256,
        "normalized_prompt_sha256": normalized_sha256,
        "audit_implementation_sha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "scan_bundle_sha256": scan_bundle.artifact_sha256,
        "container_image_digest": expected_container_image_digest,
        "methods": sorted(receipt.method for receipt in audit.methods),
        "method_receipts_sha256": canonical_sha256(
            [
                receipt.model_dump(mode="json", by_alias=True)
                for receipt in sorted(audit.methods, key=lambda item: item.method)
            ]
        ),
        "hit_count": len(audit.hits),
        "hit_records_sha256": canonical_sha256(
            [hit.model_dump(mode="json", by_alias=True) for hit in audit.hits]
        ),
        "auditor_reviewer_id_sha256": hashlib.sha256(
            audit.auditor_reviewer_id.encode("utf-8")
        ).hexdigest(),
        "status": "verified",
    }
    return _seal_receipt(payload)


def task_evidence_root_sha256(
    *,
    task_record_sha256: str,
    candidate_record_sha256: str,
    review_history_sha256: str,
    validator_contract_sha256: str,
    contamination_audit_sha256: str,
    validator_receipt_sha256: str,
    contamination_receipt_sha256: str,
    validator_review_event_sha256: str,
    contamination_review_event_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": TASK_EVIDENCE_ROOT_SCHEMA_VERSION,
            "task_record_sha256": task_record_sha256,
            "candidate_record_sha256": candidate_record_sha256,
            "review_history_sha256": review_history_sha256,
            "validator_contract_sha256": validator_contract_sha256,
            "contamination_audit_sha256": contamination_audit_sha256,
            "validator_receipt_sha256": validator_receipt_sha256,
            "contamination_receipt_sha256": contamination_receipt_sha256,
            "validator_review_event_sha256": validator_review_event_sha256,
            "contamination_review_event_sha256": contamination_review_event_sha256,
        }
    )


def task_evidence_review_sha256(
    *,
    candidate_id: str,
    candidate_record_sha256: str,
    task_public_id: str,
    reviewer_id: str,
    evidence_type: Literal["validator_contract", "contamination_audit"],
    artifact_sha256: str,
    verification_receipt_sha256: str,
    review: dict[str, object],
) -> str:
    """Seal the human inspection separately from the machine replay receipt."""

    return canonical_sha256(
        {
            "schema_version": TASK_EVIDENCE_REVIEW_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "candidate_record_sha256": candidate_record_sha256,
            "task_public_id": task_public_id,
            "reviewer_id": reviewer_id,
            "evidence_type": evidence_type,
            "artifact_sha256": artifact_sha256,
            "verification_receipt_sha256": verification_receipt_sha256,
            "review": review,
        }
    )
