from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .budget_policy import canonical_sha256
from .models import ProviderAccountAuthorization, ProviderAccountBudget


def as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def account_authorization(
    session: Session,
    ledger: ProviderAccountBudget,
    *,
    envelope_sha256: str | None = None,
    active_only: bool = True,
    for_update: bool = False,
) -> ProviderAccountAuthorization | None:
    query = select(ProviderAccountAuthorization).where(
        ProviderAccountAuthorization.provider_account_budget_id == ledger.id,
        ProviderAccountAuthorization.execution_backend == ledger.execution_backend,
        ProviderAccountAuthorization.account_scope_sha256 == ledger.account_scope_sha256,
    )
    if envelope_sha256 is not None:
        query = query.where(
            ProviderAccountAuthorization.authorization_envelope_sha256 == envelope_sha256
        )
    if active_only:
        query = query.where(ProviderAccountAuthorization.status == "active")
    query = query.order_by(ProviderAccountAuthorization.created_at.desc())
    if for_update:
        query = query.with_for_update()
    return session.scalar(query.limit(1))


def account_authorization_valid(
    ledger: ProviderAccountBudget,
    authorization: ProviderAccountAuthorization | None,
    *,
    signing_secret: str,
    verification_keys: Mapping[str, str] | None = None,
    now: datetime | None = None,
    require_active: bool = True,
    expected_supersedes_envelope_sha256: str | None = None,
) -> bool:
    if authorization is None:
        return False
    observed = now or datetime.now(UTC)
    envelope = authorization.authorization_envelope_json
    common_envelope_keys = {
        "schema_version",
        "provider_account_budget_id",
        "execution_backend",
        "currency",
        "budget_cap_micros",
        "account_scope_sha256",
        "authorization_reference_sha256",
        "ledger_opening_balance_sha256",
        "exposure_attestation_sha256",
        "cumulative_used_micros",
        "cumulative_reserved_micros",
        "credential_binding_sha256",
        "supersedes_authorization_envelope_sha256",
        "valid_until",
    }
    if not isinstance(envelope, dict):
        return False
    schema_version = envelope.get("schema_version")
    expected_envelope_keys = (
        common_envelope_keys | {"signing_key_id"}
        if schema_version == "flavourbench-provider-account-authorization-v4"
        else common_envelope_keys
    )
    if (
        schema_version
        not in {
            "flavourbench-provider-account-authorization-v3",
            "flavourbench-provider-account-authorization-v4",
        }
        or set(envelope) != expected_envelope_keys
    ):
        return False
    try:
        envelope_valid_until = datetime.fromisoformat(str(envelope["valid_until"]))
    except (TypeError, ValueError):
        return False
    if envelope_valid_until.tzinfo is None:
        return False
    envelope_sha256 = canonical_sha256(authorization.authorization_envelope_json)
    signature_secret = signing_secret
    if schema_version == "flavourbench-provider-account-authorization-v4":
        key_id = envelope.get("signing_key_id")
        if not isinstance(key_id, str) or verification_keys is None:
            return False
        signature_secret = verification_keys.get(key_id, "")
        if not signature_secret:
            return False
    signature_secrets = [signature_secret]
    if (
        schema_version == "flavourbench-provider-account-authorization-v3"
        and verification_keys is not None
    ):
        # V3 predates embedded key identifiers. Retained verification keys are
        # therefore tried explicitly so an append-only V3 root survives a
        # signing-key rotation without rewriting historical evidence.
        signature_secrets.extend(verification_keys.values())
    expected_hmacs = {
        hmac.new(
            candidate.encode(),
            envelope_sha256.encode(),
            hashlib.sha256,
        ).hexdigest()
        for candidate in signature_secrets
        if candidate
    }
    credential_scope = str(
        authorization.credential_binding_json.get("credential_scope_sha256", "unresolved")
    )
    active_valid = authorization.status == "active" and authorization.revoked_at is None
    revoked_valid = authorization.status == "revoked" and authorization.revoked_at is not None
    return (
        (active_valid if require_active else (active_valid or revoked_valid))
        and authorization.provider_account_budget_id == ledger.id
        and authorization.execution_backend == ledger.execution_backend
        and authorization.account_scope_sha256 == ledger.account_scope_sha256
        and envelope["schema_version"] == schema_version
        and envelope["provider_account_budget_id"] == ledger.id
        and envelope["execution_backend"] == ledger.execution_backend
        and envelope["currency"] == ledger.currency
        and envelope["budget_cap_micros"] == ledger.budget_cap_micros
        and envelope["account_scope_sha256"] == ledger.account_scope_sha256
        and envelope["authorization_reference_sha256"]
        == authorization.authorization_reference_sha256
        and envelope["ledger_opening_balance_sha256"] == ledger.opening_balance_sha256
        and envelope["exposure_attestation_sha256"] == authorization.exposure_attestation_sha256
        and envelope["cumulative_used_micros"] == authorization.authorized_used_micros
        and envelope["cumulative_reserved_micros"] == authorization.authorized_reserved_micros
        and envelope["credential_binding_sha256"] == authorization.credential_binding_sha256
        and envelope["supersedes_authorization_envelope_sha256"]
        == expected_supersedes_envelope_sha256
        and as_utc(envelope_valid_until) == as_utc(authorization.valid_until)
        and authorization.authorized_used_micros >= ledger.opening_used_micros
        and authorization.authorized_used_micros <= ledger.budget_used_micros
        and authorization.authorized_reserved_micros >= 0
        and authorization.authorized_used_micros + authorization.authorized_reserved_micros
        <= ledger.budget_cap_micros
        and canonical_sha256(authorization.exposure_attestation_json)
        == authorization.exposure_attestation_sha256
        and envelope_sha256 == authorization.authorization_envelope_sha256
        and any(
            hmac.compare_digest(
                expected_hmac,
                authorization.authorization_hmac_sha256,
            )
            for expected_hmac in expected_hmacs
        )
        and canonical_sha256(authorization.credential_binding_json)
        == authorization.credential_binding_sha256
        and credential_scope == ledger.account_scope_sha256
        and (not require_active or as_utc(authorization.valid_until) > observed)
    )


def account_authorization_chain_valid(
    session: Session,
    ledger: ProviderAccountBudget,
    authorization: ProviderAccountAuthorization | None,
    *,
    root_envelope_sha256: str,
    signing_secret: str,
    verification_keys: Mapping[str, str] | None = None,
    now: datetime | None = None,
    require_head_active: bool = True,
) -> bool:
    """Validate an active epoch and its signed predecessor chain to a season root."""

    current = authorization
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        predecessor = (
            session.get(
                ProviderAccountAuthorization,
                current.supersedes_authorization_id,
            )
            if current.supersedes_authorization_id is not None
            else None
        )
        predecessor_sha256 = (
            predecessor.authorization_envelope_sha256 if predecessor is not None else None
        )
        if not account_authorization_valid(
            ledger,
            current,
            signing_secret=signing_secret,
            verification_keys=verification_keys,
            now=now,
            require_active=(current is authorization and require_head_active),
            expected_supersedes_envelope_sha256=predecessor_sha256,
        ):
            return False
        if current.authorization_envelope_sha256 == root_envelope_sha256:
            return True
        current = predecessor
    return False
