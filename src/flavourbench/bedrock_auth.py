"""Credential-safe Amazon Bedrock client construction.

The AWS SDK owns authentication.  In particular, recent boto3/botocore
releases recognize ``AWS_BEARER_TOKEN_BEDROCK`` without the application
copying the token into a client argument, log record, or provenance artifact.
When that variable is absent, boto3 uses its standard credential provider
chain.  This module records only a non-secret authentication *mode hint*.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

from .budget_policy import provider_account_scope_sha256


class BedrockConfigurationError(ValueError):
    """The Bedrock lane is not explicitly and safely configured."""


BedrockStage = Literal["contract_smoke", "exploratory", "season"]
BedrockProfileScope = Literal["in_region", "global", "us", "eu", "apac"]


def _strict_bool(value: str | None, *, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BedrockConfigurationError(f"{name} must be a boolean")


def _money(value: str | None, *, name: str, default: str) -> Decimal:
    try:
        parsed = Decimal(value if value is not None else default)
    except (InvalidOperation, ValueError) as error:
        raise BedrockConfigurationError(f"{name} must be a decimal amount") from error
    if not parsed.is_finite() or parsed < 0:
        raise BedrockConfigurationError(f"{name} must be finite and non-negative")
    return parsed


@dataclass(frozen=True)
class BedrockLaneSettings:
    """Explicit settings for the isolated, provider-free Bedrock lane.

    No credential value is retained. ``auth_mode_hint`` says only whether the
    bearer-token environment variable was present when settings were frozen.
    """

    enabled: bool
    live_authorized: bool
    region: str
    auth_mode_hint: Literal["bedrock_bearer_token_env", "boto3_default_chain"]
    profile_scope: BedrockProfileScope
    stage: BedrockStage
    hard_cap_usd: Decimal
    contract_smoke_cap_usd: Decimal
    contract_smoke_evidence_sha256: str | None
    account_scope_sha256: str | None

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> BedrockLaneSettings:
        values = os.environ if environ is None else environ
        enabled = _strict_bool(
            values.get("FLAVOURBENCH_BEDROCK_ENABLED"),
            name="FLAVOURBENCH_BEDROCK_ENABLED",
        )
        live_authorized = _strict_bool(
            values.get("FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED"),
            name="FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED",
        )
        region = values.get("AWS_REGION", "").strip()
        if enabled and not region:
            raise BedrockConfigurationError(
                "Bedrock requires an explicit AWS_REGION; no region is inferred or defaulted"
            )
        if enabled and region.lower() in {"global", "in_region", "us", "eu", "apac"}:
            raise BedrockConfigurationError(
                "AWS_REGION must be a physical AWS region such as eu-west-1; global, us, "
                "eu, and apac are inference-profile scopes configured separately"
            )

        stage = values.get("FLAVOURBENCH_BEDROCK_STAGE", "contract_smoke").strip()
        if stage not in {"contract_smoke", "exploratory", "season"}:
            raise BedrockConfigurationError(
                "FLAVOURBENCH_BEDROCK_STAGE must be contract_smoke, exploratory, or season"
            )
        profile_scope = (
            values.get("FLAVOURBENCH_BEDROCK_PROFILE_SCOPE", "in_region").strip().lower()
        )
        if profile_scope not in {"in_region", "global", "us", "eu", "apac"}:
            raise BedrockConfigurationError(
                "FLAVOURBENCH_BEDROCK_PROFILE_SCOPE must be in_region, global, us, eu, or apac"
            )

        hard_cap_value = values.get("FLAVOURBENCH_BEDROCK_HARD_CAP_USD")
        cap_alias_value = values.get("FLAVOURBENCH_BEDROCK_CAP_USD")
        hard_cap = _money(
            hard_cap_value if hard_cap_value is not None else cap_alias_value,
            name="FLAVOURBENCH_BEDROCK_CAP_USD",
            default="0",
        )
        if hard_cap_value is not None and cap_alias_value is not None:
            explicit_hard_cap = _money(
                hard_cap_value,
                name="FLAVOURBENCH_BEDROCK_HARD_CAP_USD",
                default="0",
            )
            alias_cap = _money(
                cap_alias_value,
                name="FLAVOURBENCH_BEDROCK_CAP_USD",
                default="0",
            )
            if explicit_hard_cap != alias_cap:
                raise BedrockConfigurationError(
                    "FLAVOURBENCH_BEDROCK_CAP_USD and FLAVOURBENCH_BEDROCK_HARD_CAP_USD conflict"
                )
        smoke_cap = _money(
            values.get("FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD"),
            name="FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD",
            default="5",
        )
        if smoke_cap > hard_cap and hard_cap > 0:
            raise BedrockConfigurationError(
                "the contract-smoke cap cannot exceed the Bedrock hard cap"
            )
        if live_authorized and not enabled:
            raise BedrockConfigurationError(
                "Bedrock live authorization requires FLAVOURBENCH_BEDROCK_ENABLED=true"
            )
        if live_authorized and hard_cap <= 0:
            raise BedrockConfigurationError(
                "Bedrock live authorization requires a positive numeric cap"
            )
        smoke_evidence = values.get(
            "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_EVIDENCE_SHA256", ""
        ).strip()
        if stage != "contract_smoke" and (
            len(smoke_evidence) != 64
            or any(character not in "0123456789abcdef" for character in smoke_evidence)
        ):
            raise BedrockConfigurationError(
                "later Bedrock stages require a lowercase SHA-256 contract-smoke evidence digest"
            )
        account_scope = values.get("FLAVOURBENCH_BEDROCK_ACCOUNT_SCOPE_SHA256", "").strip()
        expected_account_scope = provider_account_scope_sha256("bedrock")
        if stage == "season" and account_scope not in {
            "",
            expected_account_scope,
        }:
            raise BedrockConfigurationError(
                "the Bedrock season stage account scope differs from the "
                "installation-wide governed authority"
            )
        if stage == "season":
            account_scope = expected_account_scope

        # Inspect only whether a non-blank bearer value exists; never retain it.
        auth_mode = (
            "bedrock_bearer_token_env"
            if bool(values.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())
            else "boto3_default_chain"
        )
        return cls(
            enabled=enabled,
            live_authorized=live_authorized,
            region=region,
            auth_mode_hint=auth_mode,
            profile_scope=profile_scope,  # type: ignore[arg-type]
            stage=stage,  # type: ignore[arg-type]
            hard_cap_usd=hard_cap,
            contract_smoke_cap_usd=smoke_cap,
            contract_smoke_evidence_sha256=smoke_evidence or None,
            account_scope_sha256=account_scope or None,
        )

    @property
    def effective_stage_cap_usd(self) -> Decimal:
        if self.stage == "contract_smoke":
            return min(self.hard_cap_usd, self.contract_smoke_cap_usd)
        return self.hard_cap_usd


class Boto3SessionLike(Protocol):
    def client(self, service_name: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class BedrockClients:
    control: Any
    runtime: Any
    region: str
    auth_mode_hint: str


def create_boto3_clients(
    settings: BedrockLaneSettings,
    *,
    session_factory: Callable[..., Boto3SessionLike] | None = None,
    client_config: Any | None = None,
) -> BedrockClients:
    """Create Bedrock clients without receiving or exposing credential values.

    The function is inert until called. Tests inject a fake session factory, so
    importing or testing the Bedrock lane never contacts AWS.
    """

    if not settings.enabled:
        raise BedrockConfigurationError("the Bedrock lane is disabled")
    if not settings.region:
        raise BedrockConfigurationError("AWS_REGION must be explicit")

    if session_factory is None:
        try:
            import boto3
        except ImportError as error:  # pragma: no cover - exercised without the extra installed
            raise BedrockConfigurationError(
                "install the optional 'bedrock' dependency to create AWS clients"
            ) from error
        session_factory = boto3.session.Session

    # Do not pass access keys, secret keys, session tokens, or the Bedrock API
    # key. The SDK reads AWS_BEARER_TOKEN_BEDROCK or its normal provider chain.
    session = session_factory(region_name=settings.region)
    client_kwargs = {"config": client_config} if client_config is not None else {}
    return BedrockClients(
        control=session.client("bedrock", **client_kwargs),
        runtime=session.client("bedrock-runtime", **client_kwargs),
        region=settings.region,
        auth_mode_hint=settings.auth_mode_hint,
    )
