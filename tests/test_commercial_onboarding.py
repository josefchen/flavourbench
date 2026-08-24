from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from flavourbench.database import engine, session_scope
from flavourbench.endpoint_contract import endpoint_contract_sha256
from flavourbench.engine import (
    _assert_commercial_external_work_authorized,
    _persist_provider_attempt,
)
from flavourbench.main import _active_controlled_release_authorization, app
from flavourbench.models import (
    Battle,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    EvaluationOrder,
    Job,
    ModelRouteRevision,
    ModelSubmission,
    OrganizationApiKey,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    Task,
)
from flavourbench.protocol_contract import build_protocol_bundle
from flavourbench.provider import ProviderAttemptEvent, ProviderError
from flavourbench.task_lifecycle import task_lifecycle_seal_sha256

SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}
MODEL_ID = "commercial-vendor/model-2026-07-21"
COMPARATOR_MODEL_ID = "commercial-vendor/comparator-2026-07-21"
ENDPOINT_DOCUMENT_SHA256 = hashlib.sha256(b"commercial endpoint document").hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(autouse=True)
def _isolate_commercial_worker_jobs():
    """Keep deliberately unexecuted commercial jobs out of later worker tests."""

    yield
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == "commercial-contract-test"))
        if season is None:
            return
        now = datetime.now(UTC)
        jobs = session.scalars(
            select(Job)
            .join(Battle, Job.battle_id == Battle.id)
            .where(Battle.season_id == season.id)
        ).all()
        for job in jobs:
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.last_error = "closed by commercial test isolation"
                job.completed_at = now


def _seed_commercial_season() -> None:
    with session_scope() as session:
        existing = session.scalar(select(Season).where(Season.slug == "commercial-contract-test"))
        if existing is not None:
            return
        now = datetime.now(UTC)
        protocol_bundle, protocol_bundle_sha256 = build_protocol_bundle(
            tool_registry_sha256="3" * 64,
            epicure_release_id="commercial-test-epicure",
            epicure_bundle_sha256="4" * 64,
            epicure_application_sha256="5" * 64,
            analysis_plan_sha256="6" * 64,
            model_smoke_registry_sha256="e" * 64,
        )
        model = CatalogModel(
            model_id=MODEL_ID,
            canonical_slug=MODEL_ID,
            name="Commercial Contract Test Model",
            family="commercial-test",
            catalog_source="organization_submission",
            open_weight=False,
            status="season_eligible",
            supports_tools=True,
            supports_structured_outputs=True,
            pricing_json={"prompt": "0.000001", "completion": "0.000002"},
            endpoint_json={
                "execution_backend": "openrouter",
                "provider": "provider-a",
                "fallbacks": False,
            },
            discovered_at=now,
            last_seen_at=now,
        )
        comparator = CatalogModel(
            model_id=COMPARATOR_MODEL_ID,
            canonical_slug=COMPARATOR_MODEL_ID,
            name="Commercial Comparator",
            family="commercial-test",
            catalog_source="frozen_manifest",
            open_weight=True,
            status="season_eligible",
            supports_tools=True,
            supports_structured_outputs=True,
            pricing_json={"prompt": "0.000001", "completion": "0.000002"},
            endpoint_json={
                "execution_backend": "openrouter",
                "provider": "provider-b",
                "fallbacks": False,
            },
            discovered_at=now,
            last_seen_at=now,
        )
        season = Season(
            slug="commercial-contract-test",
            name="Commercial contract test",
            status="draft",
            official=False,
            manifest_sha256="1" * 64,
            prompt_registry_sha256="2" * 64,
            tool_registry_sha256="3" * 64,
            epicure_release_id="commercial-test-epicure",
            epicure_bundle_sha256="4" * 64,
            epicure_application_sha256="5" * 64,
            analysis_plan_sha256="6" * 64,
            protocol_bundle_json=protocol_bundle,
            protocol_bundle_sha256=protocol_bundle_sha256,
            budget_cap_micros=10_000_000,
            frozen_at=None,
            created_at=now,
        )
        session.add_all([model, comparator, season])
        session.flush()
        supported_parameters = [
            "max_tokens",
            "response_format",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
        ]
        decoding = {"max_tokens": 1800, "temperature": 0.2}
        backend_contract = {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
        rate_card = {
            "currency": "USD",
            "promptPerToken": "0.000001",
            "completionPerToken": "0.000002",
        }
        session.add(
            SeasonModel(
                season_id=season.id,
                model_id=model.model_id,
                slot_role="closed_family",
                execution_backend="openrouter",
                provider_slug="provider-a",
                expected_actual_model_id=MODEL_ID,
                expected_actual_provider_slug="provider-a",
                supported_parameters_json=supported_parameters,
                decoding_json=decoding,
                endpoint_max_completion_tokens=1800,
                endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
                endpoint_contract_sha256=endpoint_contract_sha256(
                    model_id=MODEL_ID,
                    provider_slug="provider-a",
                    expected_actual_model_id=MODEL_ID,
                    expected_actual_provider_slug="provider-a",
                    supported_parameters=supported_parameters,
                    decoding=decoding,
                    endpoint_max_completion_tokens=1800,
                    endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
                ),
                backend_contract_json=backend_contract,
                backend_contract_sha256=_sha_json(backend_contract),
                rate_card_json=rate_card,
                rate_card_sha256=_sha_json(rate_card),
                eligible=True,
                manifest_sha256="1" * 64,
                worst_case_cost_micros=100,
                created_at=now,
            )
        )
        comparator_document_sha256 = hashlib.sha256(b"comparator endpoint").hexdigest()
        session.add(
            SeasonModel(
                season_id=season.id,
                model_id=comparator.model_id,
                slot_role="open_weight_family",
                execution_backend="openrouter",
                provider_slug="provider-b",
                expected_actual_model_id=COMPARATOR_MODEL_ID,
                expected_actual_provider_slug="provider-b",
                supported_parameters_json=supported_parameters,
                decoding_json=decoding,
                endpoint_max_completion_tokens=1800,
                endpoint_document_sha256=comparator_document_sha256,
                endpoint_contract_sha256=endpoint_contract_sha256(
                    model_id=COMPARATOR_MODEL_ID,
                    provider_slug="provider-b",
                    expected_actual_model_id=COMPARATOR_MODEL_ID,
                    expected_actual_provider_slug="provider-b",
                    supported_parameters=supported_parameters,
                    decoding=decoding,
                    endpoint_max_completion_tokens=1800,
                    endpoint_document_sha256=comparator_document_sha256,
                ),
                backend_contract_json=backend_contract,
                backend_contract_sha256=_sha_json(backend_contract),
                rate_card_json=rate_card,
                rate_card_sha256=_sha_json(rate_card),
                eligible=True,
                manifest_sha256="1" * 64,
                worst_case_cost_micros=100,
                created_at=now,
            )
        )
        task_public_id = "commercial-sealed-001"
        task_prompt = "Create a balanced sauce around a tart berry and toasted seed."
        task_prompt_sha256 = hashlib.sha256(task_prompt.encode()).hexdigest()
        authored_at = now - timedelta(seconds=1)
        candidate_record_sha256 = hashlib.sha256(b"commercial candidate").hexdigest()
        task_record_sha256 = hashlib.sha256(b"commercial task record").hexdigest()
        task_evidence_root_sha256 = hashlib.sha256(b"commercial task evidence").hexdigest()
        lifecycle_seal_sha256 = task_lifecycle_seal_sha256(
            task_public_id=task_public_id,
            task_revision=1,
            candidate_record_sha256=candidate_record_sha256,
            task_record_sha256=task_record_sha256,
            task_evidence_root_sha256=task_evidence_root_sha256,
            authored_at=authored_at,
            sealed_at=now,
        )
        session.flush()
        task = Task(
            public_id=task_public_id,
            season_id=season.id,
            family="composition",
            prompt=task_prompt,
            prompt_sha256=task_prompt_sha256,
            revision=1,
            split="scored",
            review_status="frozen",
            provenance_json={
                "commercial_test": True,
                "confirmatory_eligible": True,
                "candidate_record_sha256": candidate_record_sha256,
                "task_record_sha256": task_record_sha256,
                "task_evidence_root_sha256": task_evidence_root_sha256,
                "authored_at": authored_at.isoformat(),
                "sealed_at": now.isoformat(),
                "task_lifecycle_seal_sha256": lifecycle_seal_sha256,
            },
            created_at=now,
        )
        session.add(task)
        session.flush()
        session.add_all(
            [
                RunEvent(
                    entity_type="task",
                    entity_id=task.id,
                    event_type="confirmatory_task_authorship_recorded",
                    payload_json={
                        "public_id": task_public_id,
                        "revision": 1,
                        "candidate_record_sha256": candidate_record_sha256,
                        "authored_at": authored_at.isoformat(),
                    },
                    created_at=authored_at,
                ),
                RunEvent(
                    entity_type="task",
                    entity_id=task.id,
                    event_type="confirmatory_task_sealed",
                    payload_json={
                        "public_id": task_public_id,
                        "revision": 1,
                        "prompt_sha256": task_prompt_sha256,
                        "task_record_sha256": task_record_sha256,
                        "task_evidence_root_sha256": task_evidence_root_sha256,
                        "sealed_at": now.isoformat(),
                        "lifecycle_seal_sha256": lifecycle_seal_sha256,
                    },
                    created_at=now,
                ),
            ]
        )
        season.status = "active"
        season.official = True
        season.frozen_at = now


def _create_organization(client: TestClient, suffix: str) -> tuple[str, str]:
    created = client.post(
        "/v1/admin/organizations",
        headers=ADMIN_HEADERS,
        json={
            "slug": f"model-company-{suffix}",
            "legalName": f"Model Company {suffix} Ltd",
            "displayName": f"Model Company {suffix}",
            "idpTenantReference": f"idp-tenant-reference-{suffix}",
            "billingReference": f"billing-reference-{suffix}",
            "dataRegion": "eu",
            "retentionPolicy": {
                "privateEvidenceDays": 365,
                "rawPromptPolicy": "sealed-benchmark-contract",
            },
            "activate": True,
        },
    )
    assert created.status_code == 201, created.text
    organization_id = created.json()["organizationId"]
    issued = client.post(
        f"/v1/admin/organizations/{organization_id}/api-keys",
        headers=ADMIN_HEADERS,
        json={
            "label": "evaluation automation",
            "scopes": [
                "models:read",
                "models:submit",
                "orders:read",
                "orders:create",
                "orders:cancel",
                "bundles:read",
            ],
            "createdByPrincipalReference": f"principal-reference-{suffix}",
            "expiresAt": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
            "rateLimitProfile": "standard",
            "networkPolicy": {},
        },
    )
    assert issued.status_code == 201, issued.text
    return organization_id, issued.json()["apiKey"]


def _record_acceptance(
    client: TestClient,
    organization_id: str,
    agreement_type: str,
    *,
    evaluation_order_id: str | None = None,
    binding: dict | None = None,
) -> dict:
    payload = {
        "agreementType": agreement_type,
        "agreementVersion": "2026-07-21",
        "documentSha256": hashlib.sha256(f"document:{agreement_type}".encode()).hexdigest(),
        "externalEnvelopeReference": f"envelope-reference-{agreement_type}",
        "signatoryPrincipalReference": "authorized-signatory-reference",
        "authorityBasis": "authorized company signatory",
        "acceptedAt": datetime.now(UTC).isoformat(),
    }
    if evaluation_order_id is not None:
        payload["evaluationOrderId"] = evaluation_order_id
    if binding is not None:
        payload["binding"] = binding
    response = client.post(
        f"/v1/admin/organizations/{organization_id}/governance-acceptances",
        headers=ADMIN_HEADERS,
        json=payload,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_model_company_onboarding_is_tenant_scoped_and_idempotent() -> None:
    with TestClient(app) as client:
        _seed_commercial_season()
        organization_id, api_key = _create_organization(client, "alpha")
        other_organization_id, other_api_key = _create_organization(client, "beta")
        assert organization_id != other_organization_id
        organization_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {api_key}",
        }
        other_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {other_api_key}",
        }

        identity = client.get("/v1/org", headers=organization_headers)
        assert identity.status_code == 200, identity.text
        assert identity.json()["organizationId"] == organization_id
        assert "apiKey" not in identity.text

        forbidden_scope = client.post(
            f"/v1/admin/organizations/{organization_id}/api-keys",
            headers=ADMIN_HEADERS,
            json={
                "label": "unsafe publication credential",
                "scopes": ["publication:authorize"],
                "createdByPrincipalReference": "principal-publication-test",
                "expiresAt": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
                "networkPolicy": {},
            },
        )
        assert forbidden_scope.status_code == 422

        submission_request = {
            "displayName": "Commercial Contract Test Model",
            "publisher": "Model Company alpha Ltd",
            "requestedCanonicalModelId": MODEL_ID,
            "exactModelVersion": "2026-07-21",
            "releaseDate": "2026-07-21",
            "modelCardUri": "https://example.com/model-card",
            "modelCardSha256": hashlib.sha256(b"model card").hexdigest(),
            "licenseUri": "https://example.com/license",
            "licenseDocumentSha256": hashlib.sha256(b"license").hexdigest(),
            "capabilityClaims": {
                "text": True,
                "tools": True,
                "structuredOutput": True,
            },
            "contaminationDisclosure": {
                "flavourbenchTasksObserved": False,
                "attestationVersion": "1",
            },
            "route": {
                "routeKind": "managed_openrouter",
                "managedRouteReference": "operator-managed-route-alpha",
                "requestedModelId": MODEL_ID,
                "expectedActualModelId": MODEL_ID,
                "expectedActualProviderSlug": "provider-a",
                "supportedParameters": [
                    "max_tokens",
                    "response_format",
                    "structured_outputs",
                    "temperature",
                    "tool_choice",
                    "tools",
                ],
                "decodingBounds": {
                    "maxTokens": 1800,
                    "temperatureMaximum": 0.2,
                },
                "endpointDocumentSha256": ENDPOINT_DOCUMENT_SHA256,
                "dataPolicy": {
                    "training": "deny",
                    "retention": "deny",
                },
                "rateCard": {
                    "currency": "USD",
                    "promptPerToken": "0.000001",
                    "completionPerToken": "0.000002",
                },
            },
        }
        submission_headers = {
            **organization_headers,
            "Idempotency-Key": "submission-alpha-0001",
        }
        created = client.post(
            "/v1/org/model-submissions",
            headers=submission_headers,
            json=submission_request,
        )
        assert created.status_code == 201, created.text
        submission_id = created.json()["submissionId"]
        route_revision_id = created.json()["route"]["routeRevisionId"]
        replay = client.post(
            "/v1/org/model-submissions",
            headers=submission_headers,
            json=submission_request,
        )
        assert replay.status_code == 201
        assert replay.json()["submissionId"] == submission_id

        changed_request = json.loads(json.dumps(submission_request))
        changed_request["displayName"] = "Changed Name"
        conflict = client.post(
            "/v1/org/model-submissions",
            headers=submission_headers,
            json=changed_request,
        )
        assert conflict.status_code == 409

        secret_descriptor = json.loads(json.dumps(submission_request))
        secret_descriptor["route"]["dataPolicy"]["api_key"] = "sk-not-accepted"
        rejected_secret = client.post(
            "/v1/org/model-submissions",
            headers={
                **organization_headers,
                "Idempotency-Key": "submission-alpha-secret-rejection",
            },
            json=secret_descriptor,
        )
        assert rejected_secret.status_code == 422

        cross_tenant = client.get(
            f"/v1/org/model-submissions/{submission_id}",
            headers=other_headers,
        )
        assert cross_tenant.status_code == 404

        submitted = client.post(
            f"/v1/org/model-submissions/{submission_id}/submit",
            headers={
                **organization_headers,
                "Idempotency-Key": "submission-alpha-submit-0001",
            },
        )
        assert submitted.status_code == 200, submitted.text
        assert submitted.json()["status"] == "submitted"

        approved = client.post(
            f"/v1/admin/model-submissions/{submission_id}/decision",
            headers=ADMIN_HEADERS,
            json={
                "decision": "approve",
                "decisionReferenceSha256": hashlib.sha256(b"operator model approval").hexdigest(),
                "season": "commercial-contract-test",
            },
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "approved"
        assert approved.json()["route"]["approvalBasis"] == "frozen-season-route-contract"

        for agreement_type in (
            "service_terms",
            "acceptable_use",
            "benchmark_integrity_attestation",
        ):
            _record_acceptance(client, organization_id, agreement_type)

        order_request = {
            "modelSubmissionId": submission_id,
            "routeRevisionId": route_revision_id,
            "season": "commercial-contract-test",
            "evaluationProfileId": "private-comparative-v1",
            "requestedVisibility": "public_candidate",
            "budgetCapMicros": 1_000,
            "clientReference": "customer-purchase-order-alpha-001",
        }
        order_headers = {
            **organization_headers,
            "Idempotency-Key": "evaluation-order-alpha-0001",
        }
        order_created = client.post(
            "/v1/org/evaluation-orders",
            headers=order_headers,
            json=order_request,
        )
        assert order_created.status_code == 201, order_created.text
        order_id = order_created.json()["orderId"]
        assert order_created.json()["forecastCostMicros"] == 400
        assert order_created.json()["publicationStatus"] == "private"

        order_submitted = client.post(
            f"/v1/org/evaluation-orders/{order_id}/submit",
            headers={
                **organization_headers,
                "Idempotency-Key": "evaluation-order-alpha-submit-0001",
            },
        )
        assert order_submitted.status_code == 200, order_submitted.text
        assert order_submitted.json()["status"] == "submitted"

        missing_spend = client.post(
            f"/v1/admin/evaluation-orders/{order_id}/decision",
            headers=ADMIN_HEADERS,
            json={
                "decision": "approve",
                "decisionReferenceSha256": hashlib.sha256(b"operator order approval").hexdigest(),
                "quoteReferenceSha256": hashlib.sha256(b"quote alpha").hexdigest(),
            },
        )
        assert missing_spend.status_code == 409
        spend_acceptance = _record_acceptance(
            client,
            organization_id,
            "spend_authorization",
            evaluation_order_id=order_id,
            binding={
                "orderCardSha256": order_created.json()["orderCardSha256"],
                "budgetCapMicros": 1_000,
                "currency": "USD",
                "forecastCostMicros": 400,
                "routeRevisionId": route_revision_id,
                "seasonId": order_created.json()["seasonId"],
                "quoteReferenceSha256": hashlib.sha256(b"quote alpha").hexdigest(),
            },
        )
        order_approved = client.post(
            f"/v1/admin/evaluation-orders/{order_id}/decision",
            headers=ADMIN_HEADERS,
            json={
                "decision": "approve",
                "decisionReferenceSha256": hashlib.sha256(b"operator order approval").hexdigest(),
                "quoteReferenceSha256": hashlib.sha256(b"quote alpha").hexdigest(),
                "forecastCostMicros": 400,
            },
        )
        assert order_approved.status_code == 200, order_approved.text
        assert order_approved.json()["status"] == "approved"
        assert order_approved.json()["billingStatus"] == "authorized"
        assert order_approved.json()["publicationStatus"] == "private"

        provisioned = client.post(
            f"/v1/admin/evaluation-orders/{order_id}/provision",
            headers=ADMIN_HEADERS,
            json={
                "provisionReferenceSha256": hashlib.sha256(
                    b"operator provisioning alpha"
                ).hexdigest()
            },
        )
        assert provisioned.status_code == 201, provisioned.text
        assert provisioned.json()["orderStatus"] == "ready"
        assert provisioned.json()["commercialBinding"]["evaluation_order_id"] == order_id
        run_id = provisioned.json()["runId"]
        run_token = provisioned.json()["accessToken"]

        battle_created = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={
                **SERVICE_HEADERS,
                "Authorization": f"Bearer {run_token}",
            },
            json={
                "taskPublicId": "commercial-sealed-001",
                "expectedAssignmentOrdinal": 0,
                "clientNonce": "commercial-battle-alpha-0001",
            },
        )
        assert battle_created.status_code == 202, battle_created.text
        battle_id = battle_created.json()["battleId"]
        uplift_battle_created = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={
                **SERVICE_HEADERS,
                "Authorization": f"Bearer {run_token}",
            },
            json={
                "taskPublicId": "commercial-sealed-001",
                "expectedAssignmentOrdinal": 1,
                "clientNonce": "commercial-uplift-alpha-0001",
            },
        )
        assert uplift_battle_created.status_code == 202, uplift_battle_created.text
        uplift_battle_id = uplift_battle_created.json()["battleId"]

        with session_scope() as session:
            persisted_key = session.scalar(
                select(OrganizationApiKey).where(
                    OrganizationApiKey.organization_id == organization_id
                )
            )
            submission = session.get(ModelSubmission, submission_id)
            route = session.get(ModelRouteRevision, route_revision_id)
            order = session.get(EvaluationOrder, order_id)
            run = session.get(ControlledRun, run_id)
            assignments = session.scalars(
                select(ControlledRunAssignment)
                .where(ControlledRunAssignment.controlled_run_id == run_id)
                .order_by(ControlledRunAssignment.ordinal)
            ).all()
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle_id)
            ).all()
            uplift_arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == uplift_battle_id)
            ).all()
            battle = session.get(Battle, battle_id)
            events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type.in_(
                        {
                            "organization_api_key",
                            "model_submission",
                            "evaluation_order",
                        }
                    )
                )
            ).all()
            assert persisted_key is not None
            assert persisted_key.secret_hmac_sha256 != hashlib.sha256(api_key.encode()).hexdigest()
            assert submission is not None and route is not None and order is not None
            assert run is not None
            assert run.evaluation_order_id == order.id
            assert run.organization_id == organization_id
            assert run.route_revision_id == route.id
            assert run.endpoint_descriptor_sha256 == route.descriptor_sha256
            assert run.spend_authorization_id == spend_acceptance["acceptanceId"]
            assert run.spend_authorization_binding_sha256 == _sha_json(
                {
                    "orderCardSha256": order.order_card_sha256,
                    "budgetCapMicros": order.budget_cap_micros,
                    "currency": order.currency,
                    "forecastCostMicros": order.forecast_cost_micros,
                    "routeRevisionId": order.route_revision_id,
                    "seasonId": order.season_id,
                    "quoteReferenceSha256": order.quote_reference_sha256,
                }
            )
            assert len(assignments) == 2
            assert {assignment.track for assignment in assignments} == {
                "model_arena",
                "epicure_uplift",
            }
            assert battle is not None and battle.controlled_run_id == run.id
            assert len(arms) == 2
            submitted_arm = next(arm for arm in arms if arm.model_id == MODEL_ID)
            comparator_arm = next(arm for arm in arms if arm.model_id == COMPARATOR_MODEL_ID)
            assert submitted_arm.route_revision_id == route.id
            assert submitted_arm.endpoint_descriptor_sha256 == route.descriptor_sha256
            assert comparator_arm.route_revision_id is None
            assert comparator_arm.endpoint_descriptor_sha256 is None
            assert len(uplift_arms) == 2
            assert {arm.condition for arm in uplift_arms} == {"epicure_on", "epicure_off"}
            assert all(arm.model_id == MODEL_ID for arm in uplift_arms)
            assert all(arm.route_revision_id == route.id for arm in uplift_arms)
            assert all(
                arm.endpoint_descriptor_sha256 == route.descriptor_sha256 for arm in uplift_arms
            )
            serialized = json.dumps(
                {
                    "key": {
                        "prefix": persisted_key.key_prefix,
                        "hmac": persisted_key.secret_hmac_sha256,
                    },
                    "submission": submission.submission_payload_json,
                    "route": route.descriptor_json,
                    "order": order.order_card_json,
                    "events": [event.payload_json for event in events],
                },
                sort_keys=True,
            )
            assert api_key not in serialized
            assert "operator-managed-route-alpha" not in serialized
            assert order.publication_status == "private"

            submitted_arm_id = submitted_arm.id
            comparator_arm_id = comparator_arm.id
            route_descriptor_sha256 = route.descriptor_sha256

        if engine.dialect.name == "postgresql":
            with pytest.raises(DBAPIError), session_scope() as session:
                session.execute(
                    text(
                        "UPDATE controlled_runs "
                        "SET endpoint_descriptor_sha256 = NULL WHERE id = :run_id"
                    ),
                    {"run_id": run_id},
                )

            with pytest.raises(DBAPIError), session_scope() as session:
                session.execute(
                    text(
                        "UPDATE response_arms "
                        "SET endpoint_descriptor_sha256 = :wrong WHERE id = :arm_id"
                    ),
                    {"wrong": "f" * 64, "arm_id": submitted_arm_id},
                )

            with pytest.raises(DBAPIError), session_scope() as session:
                session.execute(
                    text(
                        "UPDATE response_arms "
                        "SET route_revision_id = :route_id, "
                        "endpoint_descriptor_sha256 = :descriptor_sha256 "
                        "WHERE id = :arm_id"
                    ),
                    {
                        "route_id": route_revision_id,
                        "descriptor_sha256": route_descriptor_sha256,
                        "arm_id": comparator_arm_id,
                    },
                )

        publication_binding = {
            "evaluationOrderId": order_id,
            "organizationId": organization_id,
            "orderCardSha256": order_created.json()["orderCardSha256"],
            "publicationScope": "controlled_run_results_and_evidence",
            "requestedVisibility": "public_candidate",
            "runCardSha256": provisioned.json()["runCardSha256"],
            "seasonId": order_created.json()["seasonId"],
        }
        publication_acceptance = _record_acceptance(
            client,
            organization_id,
            "publication_authorization",
            evaluation_order_id=order_id,
            binding=publication_binding,
        )
        run_token_release = client.post(
            f"/v1/controlled/runs/{run_id}/release-authorization",
            headers={
                **SERVICE_HEADERS,
                "Authorization": f"Bearer {run_token}",
            },
            json={
                "authorized": True,
                "publicationAcceptanceId": publication_acceptance["acceptanceId"],
                "authorizationReference": "customer-token-must-not-authorize-release",
            },
        )
        assert run_token_release.status_code == 401
        authorized_release = client.post(
            f"/v1/controlled/runs/{run_id}/release-authorization",
            headers=ADMIN_HEADERS,
            json={
                "authorized": True,
                "publicationAcceptanceId": publication_acceptance["acceptanceId"],
                "authorizationReference": "operator-validated-publication-acceptance",
            },
        )
        assert authorized_release.status_code == 200, authorized_release.text
        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            assert run is not None
            assert _active_controlled_release_authorization(session, run)

        publication_revocation = client.post(
            f"/v1/admin/governance-acceptances/{publication_acceptance['acceptanceId']}/revoke",
            headers=ADMIN_HEADERS,
            json={
                "documentSha256": hashlib.sha256(b"publication revocation").hexdigest(),
                "externalEnvelopeReference": "publication-revocation-envelope-alpha",
                "signatoryPrincipalReference": "authorized-signatory-reference",
                "authorityBasis": "authorized company signatory",
                "acceptedAt": datetime.now(UTC).isoformat(),
                "reasonCode": "customer_revoked_publication",
            },
        )
        assert publication_revocation.status_code == 201, publication_revocation.text
        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            assert run is not None
            assert not _active_controlled_release_authorization(session, run)
        revoked_release = client.post(
            f"/v1/controlled/runs/{run_id}/release-authorization",
            headers=ADMIN_HEADERS,
            json={
                "authorized": False,
                "authorizationReference": "operator-recorded-publication-revocation",
            },
        )
        assert revoked_release.status_code == 200, revoked_release.text

        revocation = client.post(
            f"/v1/admin/governance-acceptances/{spend_acceptance['acceptanceId']}/revoke",
            headers=ADMIN_HEADERS,
            json={
                "documentSha256": hashlib.sha256(b"spend revocation").hexdigest(),
                "externalEnvelopeReference": "spend-revocation-envelope-alpha",
                "signatoryPrincipalReference": "authorized-signatory-reference",
                "authorityBasis": "authorized company signatory",
                "acceptedAt": datetime.now(UTC).isoformat(),
                "reasonCode": "customer_revoked_spend",
            },
        )
        assert revocation.status_code == 201, revocation.text
        assert revocation.json()["status"] == "superseded"
        with session_scope() as session:
            battle = session.get(Battle, battle_id)
            assert battle is not None
            with pytest.raises(ProviderError, match="spend authorization"):
                _assert_commercial_external_work_authorized(session, battle)
            job = session.scalar(select(Job).where(Job.battle_id == battle_id))
            arm = session.scalar(select(ResponseArm).where(ResponseArm.battle_id == battle_id))
            assert job is not None and arm is not None
            job_id = job.id
            arm_id = arm.id
        for index, event_type in enumerate(
            ("request_started", "mcp_session_started", "mcp_call_started"),
            start=1,
        ):
            event = ProviderAttemptEvent(
                attempt_id=f"00000000-0000-0000-0000-{index:012d}",
                arm_id=arm_id,
                request_key_sha256=f"{index}" * 64,
                phase="tool" if event_type.startswith("mcp") else "final",
                attempt_index=0,
                event_type=event_type,
                payload_sha256="f" * 64,
            )
            with pytest.raises(ProviderError, match="spend authorization"):
                _persist_provider_attempt(
                    event,
                    job_id=job_id,
                    claimed_by="revoked-authority-test-worker",
                    claim_attempt=1,
                )
