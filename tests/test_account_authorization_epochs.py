from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import flavourbench.main as main_module
from flavourbench.account_authority import (
    account_authorization,
    account_authorization_chain_valid,
)
from flavourbench.budget_policy import (
    canonical_sha256,
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from flavourbench.main import (
    _rate_card_contract,
    admin_provider_account_authorization,
    admin_revoke_provider_account_authorization,
)
from flavourbench.models import (
    Base,
    CatalogModel,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
)
from flavourbench.schemas import (
    ProviderAccountAuthorizationCreate,
    ProviderAccountAuthorizationRevokeCreate,
)

SIGNING_SECRET = "account-epoch-test-signing-secret"
SIGNING_KEY_ID = "account-epoch-test-key"
VERIFICATION_KEYS = {SIGNING_KEY_ID: SIGNING_SECRET}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authorization_request(
    backend: str,
    *,
    used_micros: int,
    reserved_micros: int,
    suffix: str,
    supersedes: str | None = None,
    valid_until: datetime | None = None,
) -> ProviderAccountAuthorizationCreate:
    binding_kind = {
        "openrouter": "openrouter_account_endpoint_v1",
        "bedrock": "bedrock_control_plane_v1",
    }[backend]
    return ProviderAccountAuthorizationCreate(
        execution_backend=backend,
        budget_cap_micros=provider_account_hard_cap_micros(backend),
        opening_balance_sources=[
            {
                "source_kind": (
                    "provider_billing_export"
                    if used_micros or reserved_micros
                    else "initial_zero_balance_authorization"
                ),
                "artifact_sha256": _sha(f"exposure:{backend}:{suffix}"),
                "governed_used_micros": used_micros,
                "governed_reserved_micros": reserved_micros,
            }
        ],
        credential_binding={
            "binding_kind": binding_kind,
            "evidence_artifact_sha256": _sha(f"binding:{backend}:{suffix}"),
            "credential_scope_sha256": provider_account_scope_sha256(backend),
            "target_arn_sha256s": ([_sha(f"target:{suffix}")] if backend == "bedrock" else []),
            "observed_at": datetime.now(UTC),
        },
        authorization_reference_sha256=_sha(f"approval:{backend}:{suffix}"),
        supersedes_authorization_envelope_sha256=supersedes,
        valid_until=valid_until or datetime.now(UTC) + timedelta(days=30),
    )


@pytest.fixture
def authority_session(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'authority.sqlite3'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            budget_authorization_signing_secret=SIGNING_SECRET,
            budget_authorization_signing_key_id=SIGNING_KEY_ID,
            budget_authorization_verification_keys=VERIFICATION_KEYS,
        ),
    )
    session = Session(engine, autoflush=False, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_initial_rotation_and_revocation_preserve_permanent_ledger(
    authority_session: Session,
) -> None:
    first = admin_provider_account_authorization(
        _authorization_request(
            "openrouter",
            used_micros=0,
            reserved_micros=0,
            suffix="initial",
        ),
        authority_session,
    )
    ledger = authority_session.scalar(select(ProviderAccountBudget))
    assert ledger is not None
    ledger_id = ledger.id
    first_epoch = account_authorization(authority_session, ledger)
    assert first_epoch is not None
    assert first_epoch.authorization_envelope_sha256 == first["authorizationEnvelopeSha256"]
    assert first_epoch.exposure_attestation_sha256 == first["openingBalanceSha256"]

    ledger.budget_used_micros = 2_500
    ledger.budget_reserved_micros = 750
    authority_session.commit()
    rotated = admin_provider_account_authorization(
        _authorization_request(
            "openrouter",
            used_micros=2_500,
            reserved_micros=750,
            suffix="rotation",
            supersedes=first["authorizationEnvelopeSha256"],
        ),
        authority_session,
    )
    authority_session.refresh(ledger)
    authority_session.refresh(first_epoch)
    current = account_authorization(authority_session, ledger)
    assert current is not None
    assert ledger.id == ledger_id
    assert ledger.budget_used_micros == 2_500
    assert ledger.budget_reserved_micros == 750
    assert first_epoch.status == "revoked"
    assert current.supersedes_authorization_id == first_epoch.id
    assert current.authorization_envelope_sha256 == rotated["authorizationEnvelopeSha256"]
    assert account_authorization_chain_valid(
        authority_session,
        ledger,
        current,
        root_envelope_sha256=first["authorizationEnvelopeSha256"],
        signing_secret=SIGNING_SECRET,
        verification_keys=VERIFICATION_KEYS,
    )

    with pytest.raises(HTTPException, match="supersede the latest epoch"):
        admin_provider_account_authorization(
            _authorization_request(
                "openrouter",
                used_micros=2_500,
                reserved_micros=750,
                suffix="stale-rotation",
                supersedes=first["authorizationEnvelopeSha256"],
            ),
            authority_session,
        )
    authority_session.rollback()

    revoked = admin_revoke_provider_account_authorization(
        "openrouter",
        ProviderAccountAuthorizationRevokeCreate(
            authorization_envelope_sha256=rotated["authorizationEnvelopeSha256"],
            revocation_reference_sha256=_sha("emergency-revocation"),
        ),
        authority_session,
    )
    assert revoked["status"] == "revoked"
    assert account_authorization(authority_session, ledger) is None
    with pytest.raises(HTTPException, match="active provider account authorization"):
        admin_revoke_provider_account_authorization(
            "openrouter",
            ProviderAccountAuthorizationRevokeCreate(
                authorization_envelope_sha256=rotated[
                    "authorizationEnvelopeSha256"
                ],
                revocation_reference_sha256=_sha("repeat-revocation"),
            ),
            authority_session,
        )


def test_legacy_pending_activation_keeps_submitted_exposure_evidence(
    authority_session: Session,
) -> None:
    backend = "bedrock"
    scope = provider_account_scope_sha256(backend)
    legacy_opening = {
        "schema_version": "flavourbench-provider-opening-balance-legacy-0011",
        "governed_used_micros": 800,
        "governed_reserved_micros": 25,
    }
    legacy_binding = {"schema_version": "legacy-unverified"}
    legacy_envelope = {"schema_version": "legacy-0010"}
    ledger = ProviderAccountBudget(
        id=str(uuid.uuid4()),
        execution_backend=backend,
        currency="USD",
        status="pending_verification",
        budget_cap_micros=provider_account_hard_cap_micros(backend),
        budget_used_micros=800,
        budget_reserved_micros=25,
        opening_used_micros=800,
        opening_reserved_micros=25,
        account_scope_sha256=scope,
        authorization_reference_sha256=_sha("legacy-approval"),
        opening_balance_json=legacy_opening,
        opening_balance_sha256=canonical_sha256(legacy_opening),
        credential_binding_json=legacy_binding,
        credential_binding_sha256=canonical_sha256(legacy_binding),
        authorization_envelope_json=legacy_envelope,
        authorization_envelope_sha256=canonical_sha256(legacy_envelope),
        authorization_hmac_sha256="0" * 64,
        valid_until=datetime.now(UTC) + timedelta(days=1),
    )
    authority_session.add(ledger)
    authority_session.commit()

    result = admin_provider_account_authorization(
        _authorization_request(
            backend,
            used_micros=800,
            reserved_micros=25,
            suffix="legacy-activation",
        ),
        authority_session,
    )
    authority_session.refresh(ledger)
    epoch = account_authorization(authority_session, ledger)
    assert ledger.status == "active"
    assert epoch is not None
    assert epoch.exposure_attestation_sha256 == canonical_sha256(
        epoch.exposure_attestation_json
    )
    assert epoch.exposure_attestation_sha256 != ledger.opening_balance_sha256
    assert epoch.authorization_envelope_json["ledger_opening_balance_sha256"] == (
        ledger.opening_balance_sha256
    )
    assert epoch.authorization_envelope_json["exposure_attestation_sha256"] == (
        epoch.exposure_attestation_sha256
    )
    assert result["usedMicros"] == 800
    assert result["reservedMicros"] == 25


def test_expired_and_lower_exposure_authorizations_fail_closed(
    authority_session: Session,
) -> None:
    with pytest.raises(HTTPException, match="expiry"):
        admin_provider_account_authorization(
            _authorization_request(
                "openrouter",
                used_micros=0,
                reserved_micros=0,
                suffix="expired",
                valid_until=datetime.now(UTC) - timedelta(seconds=1),
            ),
            authority_session,
        )

    first = admin_provider_account_authorization(
        _authorization_request(
            "openrouter",
            used_micros=0,
            reserved_micros=0,
            suffix="lower-initial",
        ),
        authority_session,
    )
    ledger = authority_session.scalar(select(ProviderAccountBudget))
    assert ledger is not None
    ledger.budget_used_micros = 100
    authority_session.commit()
    with pytest.raises(HTTPException, match="preserve the permanent account ledger"):
        admin_provider_account_authorization(
            _authorization_request(
                "openrouter",
                used_micros=99,
                reserved_micros=0,
                suffix="lower-exposure",
                supersedes=first["authorizationEnvelopeSha256"],
            ),
            authority_session,
        )


def test_envelope_verifier_rejects_old_schema_and_field_substitution(
    authority_session: Session,
) -> None:
    admin_provider_account_authorization(
        _authorization_request(
            "openrouter",
            used_micros=0,
            reserved_micros=0,
            suffix="schema",
        ),
        authority_session,
    )
    ledger = authority_session.scalar(select(ProviderAccountBudget))
    assert ledger is not None
    epoch = authority_session.scalar(select(ProviderAccountAuthorization))
    assert epoch is not None
    assert account_authorization_chain_valid(
        authority_session,
        ledger,
        epoch,
        root_envelope_sha256=epoch.authorization_envelope_sha256,
        signing_secret=SIGNING_SECRET,
        verification_keys=VERIFICATION_KEYS,
    )

    original = dict(epoch.authorization_envelope_json)
    for field, replacement in (
        ("schema_version", "flavourbench-provider-account-authorization-v2"),
        ("provider_account_budget_id", str(uuid.uuid4())),
        ("currency", "EUR"),
        ("budget_cap_micros", ledger.budget_cap_micros - 1),
        ("authorization_reference_sha256", "0" * 64),
        ("credential_binding_sha256", "0" * 64),
    ):
        epoch.authorization_envelope_json = {**original, field: replacement}
        assert not account_authorization_chain_valid(
            authority_session,
            ledger,
            epoch,
            root_envelope_sha256=epoch.authorization_envelope_sha256,
            signing_secret=SIGNING_SECRET,
            verification_keys=VERIFICATION_KEYS,
        )
    epoch.authorization_envelope_json = {**original, "unexpected": True}
    assert not account_authorization_chain_valid(
        authority_session,
        ledger,
        epoch,
        root_envelope_sha256=epoch.authorization_envelope_sha256,
        signing_secret=SIGNING_SECRET,
        verification_keys=VERIFICATION_KEYS,
    )


def test_live_rate_card_reserves_cache_reasoning_and_request_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(execution_mode="live", max_tool_rounds=1),
    )
    model = CatalogModel(
        model_id="vendor/rate-test",
        canonical_slug="vendor/rate-test",
        context_length=10,
        pricing_json={},
    )
    rate_card, _digest, worst_case = _rate_card_contract(
        model,
        max_completion_tokens=5,
        endpoint_rate_card={
            "prompt_price_per_token": "0.001",
            "completion_price_per_token": "0.01",
            "request_price": "0.1",
            "internal_reasoning_price_per_token": "0.02",
            "input_cache_read_price_per_token": "0.006",
            "input_cache_write_price_per_token": "0.003",
            "input_cache_write_1h_price_per_token": "0.004",
            "image_price_per_unit": "1",
            "web_search_price_per_request": "2",
            "context_length": 10,
            "pricing_source_uri": "https://provider.example/pricing",
            "pricing_source_document_sha256": "a" * 64,
            "pricing_observed_at": datetime.now(UTC).isoformat(),
        },
    )
    assert rate_card["schema_version"] == "flavourbench-endpoint-rate-card-v3"
    assert rate_card["maximum_images_per_request"] == 0
    assert rate_card["maximum_web_searches_per_request"] == 0
    assert worst_case == 620_000

    incomplete = dict(rate_card)
    incomplete.pop("internal_reasoning_price_per_token")
    with pytest.raises(HTTPException, match="price and source envelope"):
        _rate_card_contract(
            model,
            max_completion_tokens=5,
            endpoint_rate_card=incomplete,
        )
