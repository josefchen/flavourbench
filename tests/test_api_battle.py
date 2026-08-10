from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from flavourbench.engine import run_worker_once
from flavourbench.main import app


HEADERS = {
    "X-FlavourBench-Service-Token": "test-service-token",
    "X-FlavourBench-Pseudonym": "a" * 64,
}


def test_battle_is_blind_until_idempotent_vote() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/v1/battles",
            headers=HEADERS,
            json={
                "prompt": "Design a practical dairy-free cauliflower main with clear doneness cues.",
                "category": "cookability",
                "researchConsent": False,
                "clientNonce": "stable-test-nonce-0001",
            },
        )
        assert created.status_code == 202, created.text
        battle_id = created.json()["battleId"]
        repeated = client.post(
            "/v1/battles",
            headers=HEADERS,
            json={
                "prompt": "Design a practical dairy-free cauliflower main with clear doneness cues.",
                "category": "cookability",
                "researchConsent": False,
                "clientNonce": "stable-test-nonce-0001",
            },
        )
        assert repeated.json()["battleId"] == battle_id

        queued = client.get(f"/v1/battles/{battle_id}", headers=HEADERS)
        assert queued.status_code == 200
        assert queued.json()["answers"] == []
        assert queued.json()["reveal"] is None

        assert asyncio.run(run_worker_once("pytest-worker")) is True
        completed = client.get(f"/v1/battles/{battle_id}", headers=HEADERS)
        assert completed.status_code == 200
        assert completed.json()["status"] == "complete"
        assert len(completed.json()["answers"]) == 2
        assert completed.json()["reveal"] is None
        assert "modelId" not in completed.text

        vote_headers = {**HEADERS, "Idempotency-Key": "vote-idempotency-0001"}
        first = client.post(
            f"/v1/battles/{battle_id}/votes",
            headers=vote_headers,
            json={"choice": "tie", "reasonTags": []},
        )
        assert first.status_code == 200, first.text
        assert len(first.json()["reveal"]["arms"]) == 2
        second = client.post(
            f"/v1/battles/{battle_id}/votes",
            headers=vote_headers,
            json={"choice": "tie", "reasonTags": []},
        )
        assert second.status_code == 200
        assert second.json()["voteId"] == first.json()["voteId"]


def test_other_pseudonym_cannot_poll_battle() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/v1/battles",
            headers=HEADERS,
            json={
                "prompt": "Suggest a non-alcoholic substitution for vermouth in a pan sauce.",
                "category": "substitution",
                "clientNonce": "stable-test-nonce-0002",
            },
        )
        battle_id = created.json()["battleId"]
        response = client.get(
            f"/v1/battles/{battle_id}",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": "b" * 64},
        )
        assert response.status_code == 404


def test_legacy_expert_assignment_route_is_retired_for_invited_reviewer() -> None:
    with TestClient(app) as client:
        admin_headers = {**HEADERS, "X-FlavourBench-Admin-Token": "test-admin-token"}
        invite_response = client.post(
            "/v1/admin/experts",
            headers=admin_headers,
            json={
                "reviewer_code": "chef-fixture",
                "qualified_families": ["evidence"],
                "qualification_reference": "verified-chef-fixture",
                "qualification_verified": True,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": "fixture-no-conflict",
                "consent_document_sha256": "1" * 64,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0.9,
                "compensation_reference": "fixture-volunteer-record",
            },
        )
        assert invite_response.status_code == 200, invite_response.text
        invitation = invite_response.json()["invitation"]
        created = client.post(
            "/v1/battles",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": "c" * 64},
            json={
                "prompt": "Explain how to use a high Epicure similarity score without claiming causation.",
                "category": "evidence",
                "clientNonce": "expert-assignment-nonce-0001",
            },
        )
        assert created.status_code == 202
        while asyncio.run(run_worker_once("expert-test-worker")):
            pass
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invitation}",
        }
        assignment = client.get("/v1/expert/assignments/next", headers=expert_headers)
        assert assignment.status_code == 410
        assert assignment.json()["detail"] == (
            "legacy expert assignments are retired; use versioned review sessions"
        )
