"""Offline freezing and verification of one Bedrock smoke endpoint contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .bedrock_manifest import (
    CATALOG_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    BedrockCatalogSnapshot,
    BedrockEndpointContract,
    BedrockManifestError,
    BedrockPriceContract,
    DiscoveredBedrockTarget,
    SanitizedArn,
    assert_public_catalog_safe,
    endpoint_manifest_payload,
    validate_contract_against_catalog,
)
from .bedrock_smoke_ledger import canonical_json, sha256_json

EVIDENCE_SCHEMA_VERSION = "flavourbench-bedrock-capability-price-evidence-v1"
FROZEN_SMOKE_MANIFEST_SCHEMA_VERSION = "flavourbench-bedrock-smoke-manifest-v1"


def _read_document(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise BedrockManifestError(f"contract input must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BedrockManifestError(f"contract input is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise BedrockManifestError(f"contract input must be a JSON object: {path}")
    assert_public_catalog_safe(value, path="$contract_input")
    return value, hashlib.sha256(raw).hexdigest()


def _sanitized_arn(value: object) -> SanitizedArn:
    if not isinstance(value, Mapping):
        raise BedrockManifestError("frozen ARN must be an object")
    return SanitizedArn(
        redacted=str(value.get("redacted") or ""),
        original_sha256=str(value.get("original_sha256") or ""),
    )


def load_catalog(path: str | Path) -> BedrockCatalogSnapshot:
    catalog_path = Path(path)
    document, _ = _read_document(catalog_path)
    digest = str(document.get("catalog_sha256") or "")
    payload = dict(document)
    payload.pop("catalog_sha256", None)
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION or sha256_json(payload) != digest:
        raise BedrockManifestError("catalog schema or content digest is invalid")
    if catalog_path.name != f"bedrock-catalog-{digest}.json":
        raise BedrockManifestError("catalog filename is not content-addressed")
    targets_raw = document.get("targets")
    if not isinstance(targets_raw, list):
        raise BedrockManifestError("catalog targets are invalid")
    targets: list[DiscoveredBedrockTarget] = []
    for value in targets_raw:
        if not isinstance(value, Mapping):
            raise BedrockManifestError("catalog target is invalid")
        targets.append(
            DiscoveredBedrockTarget(
                target_id=str(value.get("target_id") or ""),
                target_arn=_sanitized_arn(value.get("target_arn")),
                endpoint_kind=str(value.get("endpoint_kind") or ""),  # type: ignore[arg-type]
                foundation_model_ids=tuple(map(str, value.get("foundation_model_ids") or [])),
                foundation_model_arns=tuple(
                    _sanitized_arn(item) for item in value.get("foundation_model_arns") or []
                ),
                provider_name=str(value.get("provider_name") or ""),
                display_name=str(value.get("display_name") or ""),
                status=str(value.get("status") or ""),
                input_modalities=tuple(map(str, value.get("input_modalities") or [])),
                output_modalities=tuple(map(str, value.get("output_modalities") or [])),
                inference_types=tuple(map(str, value.get("inference_types") or [])),
            )
        )
    return BedrockCatalogSnapshot(
        schema_version=CATALOG_SCHEMA_VERSION,
        region=str(document.get("region") or ""),
        profile_scope=str(document.get("profile_scope") or ""),
        discovered_at=str(document.get("discovered_at") or ""),
        targets=tuple(targets),
        catalog_sha256=digest,
    )


def _official_aws_uri(value: object, *, field: str) -> str:
    uri = str(value or "")
    parsed = urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname not in {
        "aws.amazon.com",
        "docs.aws.amazon.com",
        "pricing.us-east-1.amazonaws.com",
    }:
        raise BedrockManifestError(f"{field} must be an official HTTPS AWS source")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BedrockManifestError(f"{field} must not contain credentials, query, or fragment")
    return uri


@dataclass(frozen=True)
class CapabilityPriceEvidence:
    path: Path
    file_sha256: str
    target_id: str
    canonical_model_id: str
    observed_at: str
    supports_converse: bool
    supports_tool_use: bool
    supports_structured_output: bool
    supports_count_tokens: bool
    capability_source_uris: tuple[str, ...]
    input_per_million_usd: str
    output_per_million_usd: str
    cache_read_per_million_usd: str | None
    cache_write_per_million_usd: str | None
    price_source_uri: str
    temperature_top_p_mutually_exclusive: bool


def load_capability_price_evidence(path: str | Path) -> CapabilityPriceEvidence:
    evidence_path = Path(path)
    value, digest = _read_document(evidence_path)
    if value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise BedrockManifestError("unsupported Bedrock capability evidence schema")
    capabilities = value.get("capabilities")
    inference_constraints = value.get("inference_constraints", {})
    price = value.get("price")
    sources = value.get("capability_source_uris")
    if not isinstance(capabilities, Mapping) or not isinstance(price, Mapping):
        raise BedrockManifestError("capability and price evidence are required")
    if not isinstance(inference_constraints, Mapping):
        raise BedrockManifestError("inference constraints must be an object")
    if not isinstance(sources, list) or not sources:
        raise BedrockManifestError("at least one capability source is required")
    observed_at = str(value.get("observed_at") or "")
    if not observed_at.endswith("Z") or "T" not in observed_at:
        raise BedrockManifestError("evidence observation time must be an explicit UTC timestamp")
    return CapabilityPriceEvidence(
        path=evidence_path,
        file_sha256=digest,
        target_id=str(value.get("target_id") or ""),
        canonical_model_id=str(value.get("canonical_model_id") or ""),
        observed_at=observed_at,
        supports_converse=capabilities.get("converse") is True,
        supports_tool_use=capabilities.get("client_side_tool_use") is True,
        supports_structured_output=capabilities.get("structured_output") is True,
        supports_count_tokens=capabilities.get("count_tokens") is True,
        capability_source_uris=tuple(
            _official_aws_uri(uri, field="capability source") for uri in sources
        ),
        input_per_million_usd=str(price.get("input_per_million_usd") or ""),
        output_per_million_usd=str(price.get("output_per_million_usd") or ""),
        cache_read_per_million_usd=(
            str(price["cache_read_per_million_usd"])
            if price.get("cache_read_per_million_usd") is not None
            else None
        ),
        cache_write_per_million_usd=(
            str(price["cache_write_per_million_usd"])
            if price.get("cache_write_per_million_usd") is not None
            else None
        ),
        price_source_uri=_official_aws_uri(price.get("source_uri"), field="price source"),
        temperature_top_p_mutually_exclusive=(
            inference_constraints.get("temperature_top_p_mutually_exclusive") is True
        ),
    )


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    assert_public_catalog_safe(value, path="$frozen_bedrock_smoke_manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise BedrockManifestError("refusing to overwrite a conflicting frozen manifest")
        temporary.unlink()
    else:
        os.replace(temporary, path)
    path.chmod(0o644)


def freeze_smoke_manifest(
    *,
    catalog_path: str | Path,
    evidence_path: str | Path,
    target_id: str,
    canonical_model_id: str,
    output_directory: str | Path,
) -> Path:
    catalog = load_catalog(catalog_path)
    evidence = load_capability_price_evidence(evidence_path)
    matches = [target for target in catalog.targets if target.target_id == target_id]
    if len(matches) != 1:
        raise BedrockManifestError("the requested Bedrock target is not uniquely cataloged")
    target = matches[0]
    if evidence.target_id != target_id or evidence.canonical_model_id != canonical_model_id:
        raise BedrockManifestError("capability evidence does not bind the requested endpoint")
    if not all(
        (
            evidence.supports_converse,
            evidence.supports_tool_use,
            evidence.supports_structured_output,
            evidence.supports_count_tokens,
        )
    ):
        raise BedrockManifestError(
            "contract smoke requires affirmative Converse, tool-use, structured-output, "
            "and CountTokens evidence"
        )
    if len(target.foundation_model_ids) != 1:
        raise BedrockManifestError("contract smoke CountTokens requires one foundation model ID")
    contract = BedrockEndpointContract(
        canonical_model_id=canonical_model_id,
        bedrock_target_id=target.target_id,
        bedrock_target_arn=target.target_arn,
        endpoint_kind=target.endpoint_kind,
        expected_foundation_model_ids=target.foundation_model_ids,
        destination_model_arns=target.foundation_model_arns,
        region=catalog.region,
        profile_scope=catalog.profile_scope,  # type: ignore[arg-type]
        supports_converse=evidence.supports_converse,
        supports_tool_use=evidence.supports_tool_use,
        supports_structured_output=evidence.supports_structured_output,
        capability_evidence_uri=f"artifact://{evidence.path.name}",
        capability_evidence_sha256=evidence.file_sha256,
        price=BedrockPriceContract(
            input_per_million_usd=evidence.input_per_million_usd,
            output_per_million_usd=evidence.output_per_million_usd,
            cache_read_per_million_usd=evidence.cache_read_per_million_usd,
            cache_write_per_million_usd=evidence.cache_write_per_million_usd,
            source_uri=evidence.price_source_uri,
            observed_at=evidence.observed_at,
        ),
        openrouter_fallback_model_id=canonical_model_id,
        season_eligible=False,
        temperature_top_p_mutually_exclusive=(evidence.temperature_top_p_mutually_exclusive),
    )
    validate_contract_against_catalog(contract, catalog)
    endpoint_payload = endpoint_manifest_payload(catalog, [contract])
    payload = {
        "schema_version": FROZEN_SMOKE_MANIFEST_SCHEMA_VERSION,
        "endpoint_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "catalog_filename": Path(catalog_path).name,
        "catalog_sha256": catalog.catalog_sha256,
        "capability_price_evidence_filename": evidence.path.name,
        "capability_price_evidence_sha256": evidence.file_sha256,
        "capability_source_uris": list(evidence.capability_source_uris),
        "count_tokens_model_id": target.foundation_model_ids[0],
        "count_tokens_supported": True,
        "endpoint_manifest": endpoint_payload,
        "rank_eligible": False,
        "official": False,
    }
    digest = sha256_json(payload)
    document = {**payload, "manifest_sha256": digest}
    path = Path(output_directory) / f"bedrock-smoke-manifest-{digest}.json"
    _atomic_write(path, document)
    return path


@dataclass(frozen=True)
class LoadedSmokeContract:
    manifest_path: Path
    manifest_sha256: str
    evidence_path: Path
    evidence_sha256: str
    catalog: BedrockCatalogSnapshot
    contract: BedrockEndpointContract
    document: Mapping[str, Any]


def _contract_from_payload(value: Mapping[str, Any]) -> BedrockEndpointContract:
    price = value.get("price")
    if not isinstance(price, Mapping):
        raise BedrockManifestError("endpoint contract has no price object")
    return BedrockEndpointContract(
        canonical_model_id=str(value.get("canonical_model_id") or ""),
        bedrock_target_id=str(value.get("bedrock_target_id") or ""),
        bedrock_target_arn=_sanitized_arn(value.get("bedrock_target_arn")),
        endpoint_kind=str(value.get("endpoint_kind") or ""),  # type: ignore[arg-type]
        expected_foundation_model_ids=tuple(
            map(str, value.get("expected_foundation_model_ids") or [])
        ),
        destination_model_arns=tuple(
            _sanitized_arn(item) for item in value.get("destination_model_arns") or []
        ),
        region=str(value.get("region") or ""),
        profile_scope=str(value.get("profile_scope") or ""),  # type: ignore[arg-type]
        supports_converse=value.get("supports_converse") is True,
        supports_tool_use=value.get("supports_tool_use") is True,
        supports_structured_output=value.get("supports_structured_output") is True,
        capability_evidence_uri=str(value.get("capability_evidence_uri") or ""),
        capability_evidence_sha256=str(value.get("capability_evidence_sha256") or ""),
        price=BedrockPriceContract(
            input_per_million_usd=str(price.get("input_per_million_usd") or ""),
            output_per_million_usd=str(price.get("output_per_million_usd") or ""),
            cache_read_per_million_usd=(
                str(price["cache_read_per_million_usd"])
                if price.get("cache_read_per_million_usd") is not None
                else None
            ),
            cache_write_per_million_usd=(
                str(price["cache_write_per_million_usd"])
                if price.get("cache_write_per_million_usd") is not None
                else None
            ),
            source_uri=str(price.get("source_uri") or ""),
            observed_at=str(price.get("observed_at") or ""),
        ),
        openrouter_fallback_model_id=(
            str(value["openrouter_fallback_model_id"])
            if value.get("openrouter_fallback_model_id") is not None
            else None
        ),
        season_eligible=value.get("season_eligible") is True,
        temperature_top_p_mutually_exclusive=(
            value.get("temperature_top_p_mutually_exclusive") is True
        ),
    )


def parse_bedrock_endpoint_contract(
    value: Mapping[str, Any],
) -> BedrockEndpointContract:
    """Validate and reconstruct a frozen Bedrock endpoint contract payload."""

    contract = _contract_from_payload(value)
    if contract.payload() != dict(value):
        raise BedrockManifestError("Bedrock endpoint contract is not in canonical normalized form")
    return contract


def load_smoke_contract(
    *,
    manifest_path: str | Path,
    catalog_path: str | Path,
    evidence_path: str | Path,
    expected_manifest_sha256: str,
) -> LoadedSmokeContract:
    path = Path(manifest_path)
    document, _ = _read_document(path)
    digest = str(document.get("manifest_sha256") or "")
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    if (
        document.get("schema_version") != FROZEN_SMOKE_MANIFEST_SCHEMA_VERSION
        or digest != sha256_json(payload)
        or digest != expected_manifest_sha256
        or path.name != f"bedrock-smoke-manifest-{digest}.json"
    ):
        raise BedrockManifestError("frozen Bedrock smoke manifest identity is invalid")
    catalog = load_catalog(catalog_path)
    if (
        document.get("catalog_sha256") != catalog.catalog_sha256
        or document.get("catalog_filename") != Path(catalog_path).name
    ):
        raise BedrockManifestError("smoke manifest does not link the supplied catalog")
    evidence = load_capability_price_evidence(evidence_path)
    if (
        document.get("capability_price_evidence_sha256") != evidence.file_sha256
        or document.get("capability_price_evidence_filename") != evidence.path.name
    ):
        raise BedrockManifestError("smoke manifest does not link the supplied capability evidence")
    endpoint_manifest = document.get("endpoint_manifest")
    if not isinstance(endpoint_manifest, Mapping):
        raise BedrockManifestError("smoke endpoint manifest is absent")
    contracts = endpoint_manifest.get("contracts")
    if (
        not isinstance(contracts, list)
        or len(contracts) != 1
        or not isinstance(contracts[0], Mapping)
    ):
        raise BedrockManifestError("smoke manifest must freeze exactly one endpoint")
    contract = _contract_from_payload(contracts[0])
    if contract.season_eligible or document.get("rank_eligible") is not False:
        raise BedrockManifestError("contract smoke must remain rank-ineligible")
    if contract.capability_evidence_sha256 != evidence.file_sha256:
        raise BedrockManifestError("endpoint capability digest differs from its linked evidence")
    if (
        contract.temperature_top_p_mutually_exclusive
        != evidence.temperature_top_p_mutually_exclusive
    ):
        raise BedrockManifestError("endpoint sampling constraint differs from its linked evidence")
    if contract.price.sha256 != contracts[0].get("price", {}).get("price_sha256"):
        raise BedrockManifestError("endpoint frozen price digest is invalid")
    validate_contract_against_catalog(contract, catalog)
    return LoadedSmokeContract(
        manifest_path=path,
        manifest_sha256=digest,
        evidence_path=Path(evidence_path),
        evidence_sha256=evidence.file_sha256,
        catalog=catalog,
        contract=contract,
        document=document,
    )
