"""Bedrock control-plane discovery and content-addressed endpoint manifests.

Discovery uses only Amazon Bedrock control-plane list operations. It never
calls ``Converse`` or another inference operation. Capability declarations are
kept separate because the catalog APIs do not prove tool-use or structured-
output conformance for a particular model/endpoint combination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients

CATALOG_SCHEMA_VERSION = "flavourbench-bedrock-catalog-v2"
MANIFEST_SCHEMA_VERSION = "flavourbench-bedrock-endpoint-manifest-v2"
DISCOVERY_CONFIRMATION = "DISCOVER_BEDROCK_CATALOG_WITHOUT_INFERENCE"

EndpointKind = Literal["foundation_model", "inference_profile", "provisioned_throughput"]


class BedrockManifestError(ValueError):
    """A discovered target or frozen binding is ambiguous or unsafe."""


_AWS_ACCOUNT_ID = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")
_AWS_ACCESS_KEY_ID = re.compile(
    r"(?<![A-Z0-9])(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}"
    r"(?![A-Z0-9])"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SECRET_QUERY = re.compile(r"(?i)[?&](?:api[_-]?key|token|secret|password)=[^&#\s]+")
_URI_USERINFO = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@")
_CONTENT_ADDRESSED_FILENAME = re.compile(r"^(?P<prefix>.*-)[0-9a-f]{64}(?P<suffix>\.json)?$")
_OFFICIAL_PRICE_LIST_URI_PREFIX = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "accountid",
        "apikey",
        "authorization",
        "awsaccesskeyid",
        "awsaccesstoken",
        "awsbearertokenbedrock",
        "awssecretaccesskey",
        "bearertoken",
        "cookie",
        "credential",
        "headers",
        "password",
        "rawarn",
        "responsemetadata",
        "secret",
        "sessiontoken",
    }
)
_PUBLIC_DIGEST_KEYS = frozenset({"runkey", "armid", "reservationid"})


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def assert_public_catalog_safe(value: object, *, path: str = "$catalog") -> None:
    """Fail closed if public catalog material resembles identity or credentials.

    Error messages identify only the field path and leak category, never the
    rejected value.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            normalized_key = _normalized_key(key)
            if normalized_key in _FORBIDDEN_PUBLIC_KEYS:
                raise BedrockManifestError(
                    f"public catalog contains a forbidden credential/identity field at {item_path}"
                )
            if normalized_key.endswith("sha256") or normalized_key in _PUBLIC_DIGEST_KEYS:
                if item is None:
                    continue
                if (
                    not isinstance(item, str)
                    or len(item) != 64
                    or any(character not in "0123456789abcdef" for character in item)
                ):
                    raise BedrockManifestError(
                        f"public catalog contains an invalid SHA-256 at {item_path}"
                    )
                continue
            if normalized_key.endswith("sha256s"):
                if not isinstance(item, Sequence) or isinstance(item, str | bytes):
                    raise BedrockManifestError(
                        f"public catalog contains an invalid SHA-256 list at {item_path}"
                    )
                if any(
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for digest in item
                ):
                    raise BedrockManifestError(
                        f"public catalog contains an invalid SHA-256 list at {item_path}"
                    )
                continue
            assert_public_catalog_safe(item, path=item_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            assert_public_catalog_safe(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    # A SHA-256 suffix can legitimately contain twelve decimal digits. Scan
    # the human-controlled filename prefix while excluding only the verified
    # 64-hex content-address component.
    content_addressed = _CONTENT_ADDRESSED_FILENAME.fullmatch(value)
    identity_scan_value = (
        (content_addressed.group("prefix") + (content_addressed.group("suffix") or ""))
        if content_addressed
        else value
    )
    if not value.startswith(_OFFICIAL_PRICE_LIST_URI_PREFIX) and _AWS_ACCOUNT_ID.search(
        identity_scan_value
    ):
        raise BedrockManifestError(f"public catalog contains an AWS account ID at {path}")
    if _AWS_ACCESS_KEY_ID.search(identity_scan_value):
        raise BedrockManifestError(f"public catalog contains an AWS access-key ID at {path}")
    if _BEARER_CREDENTIAL.search(value):
        raise BedrockManifestError(f"public catalog contains a bearer credential at {path}")
    if _SECRET_QUERY.search(value) or _URI_USERINFO.search(value):
        raise BedrockManifestError(f"public catalog contains a credential-bearing URI at {path}")


@dataclass(frozen=True)
class SanitizedArn:
    redacted: str
    original_sha256: str

    def __post_init__(self) -> None:
        if not self.redacted.startswith("arn:"):
            raise BedrockManifestError("a sanitized AWS ARN must retain its ARN structure")
        if len(self.original_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.original_sha256
        ):
            raise BedrockManifestError("a sanitized AWS ARN requires a lowercase SHA-256")
        assert_public_catalog_safe(asdict(self), path="$arn")


def sanitized_arn(raw_arn: object) -> SanitizedArn:
    """Immediately discard an ARN's raw account identity after hashing it."""

    raw = str(raw_arn or "")
    if not raw.startswith("arn:"):
        raise BedrockManifestError("Bedrock discovery returned an invalid ARN")
    return SanitizedArn(
        redacted=_AWS_ACCOUNT_ID.sub("<account-redacted>", raw),
        original_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


class BedrockControlClient(Protocol):
    def list_foundation_models(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def list_inference_profiles(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def list_provisioned_model_throughputs(self, **kwargs: Any) -> Mapping[str, Any]: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _foundation_id_from_arn(arn: object) -> str:
    value = str(arn or "")
    marker = "foundation-model/"
    return value.split(marker, 1)[1] if marker in value else value


def _money_text(value: object, *, field: str) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BedrockManifestError(f"{field} must be a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise BedrockManifestError(f"{field} must be finite and non-negative")
    rendered = format(parsed, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass(frozen=True)
class DiscoveredBedrockTarget:
    target_id: str
    target_arn: SanitizedArn
    endpoint_kind: EndpointKind
    foundation_model_ids: tuple[str, ...]
    foundation_model_arns: tuple[SanitizedArn, ...]
    provider_name: str
    display_name: str
    status: str
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    inference_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_id:
            raise BedrockManifestError("a discovered Bedrock target requires an ID")
        if not self.foundation_model_ids:
            raise BedrockManifestError(
                f"Bedrock target {self.target_id!r} has no attributable foundation model"
            )
        if not self.foundation_model_arns:
            raise BedrockManifestError(
                f"Bedrock target {self.target_id!r} has no attributable destination ARN"
            )
        assert_public_catalog_safe(asdict(self), path="$target")


@dataclass(frozen=True)
class BedrockCatalogSnapshot:
    schema_version: str
    region: str
    profile_scope: str
    discovered_at: str
    targets: tuple[DiscoveredBedrockTarget, ...]
    catalog_sha256: str

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "region": self.region,
            "profile_scope": self.profile_scope,
            "discovered_at": self.discovered_at,
            "targets": [asdict(target) for target in self.targets],
        }
        assert_public_catalog_safe(payload)
        return payload


class BedrockCatalogDiscoverer:
    """Normalize foundation models, inference profiles, and provisioned targets."""

    def __init__(
        self,
        client: BedrockControlClient,
        *,
        region: str,
        profile_scope: str = "in_region",
    ) -> None:
        if not region:
            raise BedrockManifestError("catalog discovery requires an explicit region")
        if profile_scope not in {"in_region", "global", "us", "eu", "apac"}:
            raise BedrockManifestError("catalog discovery received an invalid profile scope")
        self.client = client
        self.region = region
        self.profile_scope = profile_scope

    def _pages(self, operation: str, result_key: str) -> list[Mapping[str, Any]]:
        method = getattr(self.client, operation)
        items: list[Mapping[str, Any]] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            kwargs = {"nextToken": token} if token else {}
            response = method(**kwargs)
            raw_items = response.get(result_key, [])
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, str | bytes):
                raise BedrockManifestError(f"{operation} returned an invalid {result_key}")
            items.extend(item for item in raw_items if isinstance(item, Mapping))
            next_token = response.get("nextToken")
            if not isinstance(next_token, str) or not next_token:
                return items
            if next_token in seen_tokens:
                raise BedrockManifestError(f"{operation} returned a repeated nextToken")
            seen_tokens.add(next_token)
            token = next_token

    def discover(self, *, discovered_at: str | None = None) -> BedrockCatalogSnapshot:
        targets: list[DiscoveredBedrockTarget] = []
        foundation = self.client.list_foundation_models()
        models = foundation.get("modelSummaries", [])
        if not isinstance(models, Sequence) or isinstance(models, str | bytes):
            raise BedrockManifestError("list_foundation_models returned invalid summaries")
        for model in models:
            if not isinstance(model, Mapping):
                continue
            model_id = str(model.get("modelId") or "")
            lifecycle = model.get("modelLifecycle")
            status = (
                str(lifecycle.get("status") or "UNKNOWN")
                if isinstance(lifecycle, Mapping)
                else "UNKNOWN"
            )
            targets.append(
                DiscoveredBedrockTarget(
                    target_id=model_id,
                    target_arn=sanitized_arn(model.get("modelArn")),
                    endpoint_kind="foundation_model",
                    foundation_model_ids=(model_id,),
                    foundation_model_arns=(sanitized_arn(model.get("modelArn")),),
                    provider_name=str(model.get("providerName") or ""),
                    display_name=str(model.get("modelName") or model_id),
                    status=status,
                    input_modalities=tuple(sorted(map(str, model.get("inputModalities") or []))),
                    output_modalities=tuple(sorted(map(str, model.get("outputModalities") or []))),
                    inference_types=tuple(
                        sorted(map(str, model.get("inferenceTypesSupported") or []))
                    ),
                )
            )

        for profile in self._pages("list_inference_profiles", "inferenceProfileSummaries"):
            profile_id = str(profile.get("inferenceProfileId") or "")
            model_refs = profile.get("models") or []
            foundation_ids = tuple(
                sorted(
                    {
                        _foundation_id_from_arn(item.get("modelArn"))
                        for item in model_refs
                        if isinstance(item, Mapping) and item.get("modelArn")
                    }
                )
            )
            foundation_arns = tuple(
                sorted(
                    {
                        sanitized_arn(item.get("modelArn"))
                        for item in model_refs
                        if isinstance(item, Mapping) and item.get("modelArn")
                    },
                    key=lambda item: (item.redacted, item.original_sha256),
                )
            )
            targets.append(
                DiscoveredBedrockTarget(
                    target_id=profile_id,
                    target_arn=sanitized_arn(profile.get("inferenceProfileArn")),
                    endpoint_kind="inference_profile",
                    foundation_model_ids=foundation_ids,
                    foundation_model_arns=foundation_arns,
                    provider_name="Amazon Bedrock",
                    display_name=str(profile.get("inferenceProfileName") or profile_id),
                    status=str(profile.get("status") or "UNKNOWN"),
                    inference_types=(str(profile.get("type") or "UNKNOWN"),),
                )
            )

        for provisioned in self._pages(
            "list_provisioned_model_throughputs", "provisionedModelSummaries"
        ):
            target_id = str(provisioned.get("provisionedModelName") or "")
            foundation_id = _foundation_id_from_arn(
                provisioned.get("foundationModelArn") or provisioned.get("modelArn")
            )
            targets.append(
                DiscoveredBedrockTarget(
                    target_id=target_id,
                    target_arn=sanitized_arn(provisioned.get("provisionedModelArn")),
                    endpoint_kind="provisioned_throughput",
                    foundation_model_ids=(foundation_id,),
                    foundation_model_arns=(
                        sanitized_arn(
                            provisioned.get("foundationModelArn") or provisioned.get("modelArn")
                        ),
                    ),
                    provider_name="Amazon Bedrock",
                    display_name=str(provisioned.get("provisionedModelName") or target_id),
                    status=str(provisioned.get("status") or "UNKNOWN"),
                    inference_types=("PROVISIONED",),
                )
            )

        unique: dict[tuple[str, str], DiscoveredBedrockTarget] = {}
        for target in targets:
            key = (target.endpoint_kind, target.target_id)
            if key in unique:
                raise BedrockManifestError(f"duplicate Bedrock discovery target: {key}")
            unique[key] = target
        ordered = tuple(
            sorted(unique.values(), key=lambda item: (item.endpoint_kind, item.target_id))
        )
        timestamp = discovered_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        payload = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "region": self.region,
            "profile_scope": self.profile_scope,
            "discovered_at": timestamp,
            "targets": [asdict(target) for target in ordered],
        }
        assert_public_catalog_safe(payload)
        return BedrockCatalogSnapshot(
            schema_version=CATALOG_SCHEMA_VERSION,
            region=self.region,
            profile_scope=self.profile_scope,
            discovered_at=timestamp,
            targets=ordered,
            catalog_sha256=_sha256(payload),
        )


@dataclass(frozen=True)
class BedrockPriceContract:
    input_per_million_usd: str
    output_per_million_usd: str
    cache_read_per_million_usd: str | None
    cache_write_per_million_usd: str | None
    source_uri: str
    observed_at: str

    def normalized(self) -> dict[str, Any]:
        if not self.source_uri or not self.observed_at:
            raise BedrockManifestError("a frozen Bedrock price needs source and observation time")
        return {
            "input_per_million_usd": _money_text(
                self.input_per_million_usd, field="input token price"
            ),
            "output_per_million_usd": _money_text(
                self.output_per_million_usd, field="output token price"
            ),
            "cache_read_per_million_usd": (
                _money_text(self.cache_read_per_million_usd, field="cache-read price")
                if self.cache_read_per_million_usd is not None
                else None
            ),
            "cache_write_per_million_usd": (
                _money_text(self.cache_write_per_million_usd, field="cache-write price")
                if self.cache_write_per_million_usd is not None
                else None
            ),
            "source_uri": self.source_uri,
            "observed_at": self.observed_at,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.normalized())


@dataclass(frozen=True)
class BedrockEndpointContract:
    canonical_model_id: str
    bedrock_target_id: str
    bedrock_target_arn: SanitizedArn
    endpoint_kind: EndpointKind
    expected_foundation_model_ids: tuple[str, ...]
    destination_model_arns: tuple[SanitizedArn, ...]
    region: str
    profile_scope: Literal["in_region", "global", "us", "eu", "apac"]
    supports_converse: bool
    supports_tool_use: bool
    supports_structured_output: bool
    capability_evidence_uri: str
    capability_evidence_sha256: str
    price: BedrockPriceContract
    openrouter_fallback_model_id: str | None = None
    season_eligible: bool = False
    temperature_top_p_mutually_exclusive: bool = False

    def __post_init__(self) -> None:
        if not self.canonical_model_id or not self.bedrock_target_id or not self.region:
            raise BedrockManifestError("Bedrock endpoint identity must be complete")
        if not self.expected_foundation_model_ids:
            raise BedrockManifestError("Bedrock endpoint must freeze its foundation-model identity")
        if not self.destination_model_arns:
            raise BedrockManifestError("Bedrock endpoint must freeze destination model ARNs")
        if self.endpoint_kind != "inference_profile" and self.profile_scope != "in_region":
            raise BedrockManifestError(
                "cross-region profile scope requires an inference-profile target"
            )
        if self.openrouter_fallback_model_id not in {None, self.canonical_model_id}:
            raise BedrockManifestError(
                "OpenRouter fallback must use exactly the same canonical model ID"
            )
        if self.season_eligible and not all(
            (self.supports_converse, self.supports_tool_use, self.supports_structured_output)
        ):
            raise BedrockManifestError(
                "a season-eligible Bedrock endpoint must attest Converse, tools, "
                "and structured output"
            )
        if self.season_eligible and self.endpoint_kind == "provisioned_throughput":
            raise BedrockManifestError(
                "provisioned throughput is not season-eligible until hourly "
                "commitment accounting is governed"
            )
        if not self.capability_evidence_uri or len(self.capability_evidence_sha256) != 64:
            raise BedrockManifestError("Bedrock capability evidence must be content-addressed")
        assert_public_catalog_safe(self.payload(), path="$endpoint_contract")

    def payload(self) -> dict[str, Any]:
        payload = {
            "canonical_model_id": self.canonical_model_id,
            "bedrock_target_id": self.bedrock_target_id,
            "bedrock_target_arn": asdict(self.bedrock_target_arn),
            "endpoint_kind": self.endpoint_kind,
            "expected_foundation_model_ids": sorted(self.expected_foundation_model_ids),
            "destination_model_arns": [
                asdict(item)
                for item in sorted(
                    self.destination_model_arns,
                    key=lambda value: (value.redacted, value.original_sha256),
                )
            ],
            "region": self.region,
            "profile_scope": self.profile_scope,
            "profile_scope_sha256": self.profile_scope_sha256,
            "supports_converse": self.supports_converse,
            "supports_tool_use": self.supports_tool_use,
            "supports_structured_output": self.supports_structured_output,
            "capability_evidence_uri": self.capability_evidence_uri,
            "capability_evidence_sha256": self.capability_evidence_sha256,
            "price": {**self.price.normalized(), "price_sha256": self.price.sha256},
            "openrouter_fallback_model_id": self.openrouter_fallback_model_id,
            "season_eligible": self.season_eligible,
        }
        if self.temperature_top_p_mutually_exclusive:
            payload["temperature_top_p_mutually_exclusive"] = True
        assert_public_catalog_safe(payload, path="$endpoint_contract")
        return payload

    @property
    def sha256(self) -> str:
        return _sha256(self.payload())

    @property
    def profile_scope_sha256(self) -> str:
        return _sha256(
            {
                "bedrock_target_id": self.bedrock_target_id,
                "bedrock_target_arn": asdict(self.bedrock_target_arn),
                "client_region": self.region,
                "destination_model_arns": [
                    asdict(item)
                    for item in sorted(
                        self.destination_model_arns,
                        key=lambda value: (value.redacted, value.original_sha256),
                    )
                ],
                "profile_scope": self.profile_scope,
            }
        )


def validate_contract_against_catalog(
    contract: BedrockEndpointContract,
    catalog: BedrockCatalogSnapshot,
) -> None:
    if contract.region != catalog.region:
        raise BedrockManifestError("endpoint contract and catalog regions differ")
    if contract.profile_scope != catalog.profile_scope:
        raise BedrockManifestError("endpoint contract and catalog profile scopes differ")
    matches = [
        target
        for target in catalog.targets
        if target.target_id == contract.bedrock_target_id
        and target.endpoint_kind == contract.endpoint_kind
    ]
    if len(matches) != 1:
        raise BedrockManifestError("Bedrock endpoint was not uniquely discovered")
    target = matches[0]
    if tuple(sorted(target.foundation_model_ids)) != tuple(
        sorted(contract.expected_foundation_model_ids)
    ):
        raise BedrockManifestError("discovered and frozen foundation-model identities differ")
    target_arns = sorted(
        target.foundation_model_arns,
        key=lambda value: (value.redacted, value.original_sha256),
    )
    contract_arns = sorted(
        contract.destination_model_arns,
        key=lambda value: (value.redacted, value.original_sha256),
    )
    if target_arns != contract_arns:
        raise BedrockManifestError("discovered and frozen destination model ARNs differ")
    if target.target_arn != contract.bedrock_target_arn:
        raise BedrockManifestError("discovered and frozen Bedrock target ARNs differ")
    if target.status.upper() not in {"ACTIVE", "INSERVICE", "IN_SERVICE"}:
        raise BedrockManifestError(f"Bedrock endpoint is not active: {target.status}")


def endpoint_manifest_payload(
    catalog: BedrockCatalogSnapshot,
    contracts: Sequence[BedrockEndpointContract],
) -> dict[str, Any]:
    seen: set[str] = set()
    ordered: list[BedrockEndpointContract] = []
    for contract in contracts:
        if contract.canonical_model_id in seen:
            raise BedrockManifestError("duplicate canonical model in Bedrock manifest")
        validate_contract_against_catalog(contract, catalog)
        seen.add(contract.canonical_model_id)
        ordered.append(contract)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "catalog_sha256": catalog.catalog_sha256,
        "region": catalog.region,
        "profile_scope": catalog.profile_scope,
        "rank_eligible": False,
        "official": False,
        "contracts": [
            item.payload() for item in sorted(ordered, key=lambda item: item.canonical_model_id)
        ],
    }
    assert_public_catalog_safe(payload, path="$endpoint_manifest")
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    assert_public_catalog_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _catalog_document(snapshot: BedrockCatalogSnapshot) -> dict[str, Any]:
    document = {**snapshot.payload(), "catalog_sha256": snapshot.catalog_sha256}
    assert_public_catalog_safe(document)
    return document


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or perform Bedrock control-plane catalog discovery without inference"
    )
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bedrock/catalog"))
    arguments = parser.parse_args(argv)
    settings = BedrockLaneSettings.from_environ()
    plan = {
        "operation": "bedrock_control_plane_catalog_discovery",
        "inference_calls": 0,
        "region": settings.region,
        "profile_scope": settings.profile_scope,
        "auth_mode_hint": settings.auth_mode_hint,
        "stage": settings.stage,
        "contract_smoke_evidence_sha256": settings.contract_smoke_evidence_sha256,
        "effective_stage_cap_usd": str(settings.effective_stage_cap_usd),
        "enabled": settings.enabled,
    }
    assert_public_catalog_safe(plan, path="$preflight_plan")
    if not arguments.discover:
        print(json.dumps(plan, sort_keys=True))
        return 0
    if arguments.confirm != DISCOVERY_CONFIRMATION:
        raise SystemExit(f"--discover requires --confirm {DISCOVERY_CONFIRMATION}")

    clients = create_boto3_clients(settings)
    snapshot = BedrockCatalogDiscoverer(
        clients.control,
        region=settings.region,
        profile_scope=settings.profile_scope,
    ).discover()
    document = _catalog_document(snapshot)
    assert_public_catalog_safe(str(arguments.output_dir), path="$preflight_output_dir")
    path = arguments.output_dir / f"bedrock-catalog-{snapshot.catalog_sha256}.json"
    _atomic_write(path, document)
    print(
        json.dumps(
            {
                **plan,
                "catalog_sha256": snapshot.catalog_sha256,
                "target_count": len(snapshot.targets),
                "output": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
