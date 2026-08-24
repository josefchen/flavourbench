from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from flavourbench import reasoning_effort_full_study_executor_v1 as executor
from flavourbench import reasoning_effort_full_study_v1 as study
from flavourbench import reasoning_effort_import_incident_recovery_v1 as recovery

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT = REPO_ROOT / "flavourbench/artifacts/season1/current-quality-run"
PLAN = CURRENT / (
    "reasoning-effort-task-waves-v3/plan/"
    "reasoning-effort-task-wave-plan-v2-"
    "99b8f70ae81aa3a7b7e79a45bb4253cb58d26306f90ab5b9c4f09a6938f1a301.json"
)
HUMAN = CURRENT / (
    "reasoning-effort-human-protocol-v2/"
    "reasoning-effort-human-protocol-"
    "cd2a234f617158304a5eb4efed1c6e34198cd857f2de124b10dee09fdec370a8.json"
)
BOUND = CURRENT / (
    "reasoning-effort-task-waves-v3/bound-preflight/"
    "reasoning-effort-bound-admission-preflight-v2-"
    "58d509d8c9c4276ad9c497789652a9ba55a50320846b7dda5eb72853bfe25910.json"
)
RECOVERY_DIR = CURRENT / "reasoning-effort-task-waves-v3/import-incident-recovery-v1"
INCIDENT = RECOVERY_DIR / (
    "reasoning-effort-import-pipeline-incident-"
    "2385d025f33b3286ba48f36e7e493be49ce5a55a07ee94d88f757b130ae88ea3.json"
)
CONTRACT = RECOVERY_DIR / (
    "reasoning-effort-import-recovery-contract-"
    "19c2494964d4d2bb34784441f06f68d2e63cf10975af49764642e3aaa625d6c4.json"
)
RECEIPT = RECOVERY_DIR / (
    "reasoning-effort-import-recovery-receipt-"
    "ab6d988c1d63473163bb4a0ec821923e3703778f93af0c3ee1fab1a12d258eeb.json"
)


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_incident_contract_and_receipt_are_exact() -> None:
    incident = _document(INCIDENT)
    contract = _document(CONTRACT)
    receipt = _document(RECEIPT)
    assert study._artifact_ok(incident, recovery.INCIDENT_SCHEMA)
    assert study._artifact_ok(contract, recovery.CONTRACT_SCHEMA)
    assert study._artifact_ok(receipt, recovery.RECEIPT_SCHEMA)
    assert incident["study_plan_sha256"] == recovery.PLAN_SHA256
    assert incident["impact"] == {
        "scheduled_pairs": 28,
        "generated_pairs": 0,
        "provider_completion_requests": 0,
        "epicure_calls": 0,
        "actual_cost_usd": "0",
        "model_reliability_eligible": False,
        "preference_eligible": False,
        "rank_eligible": False,
    }
    assert contract["pipeline_incident"]["semantic_sha256"] == incident["artifact_sha256"]
    assert receipt["recovery_contract_sha256"] == contract["artifact_sha256"]
    assert receipt["pre_generation_failure_pairs"] == 28
    assert receipt["provider_completion_requests"] == 0
    assert receipt["epicure_calls"] == 0
    assert receipt["actual_cost_usd"] == "0"
    assert receipt["reservation_released"] is True
    assert receipt["replay_permitted"] is False
    assert receipt["rank_eligible"] is False


def test_recovery_is_network_free_and_idempotent(monkeypatch) -> None:
    plan = _document(PLAN)
    live_coordinator, live_endpoints = executor._roots(plan, REPO_ROOT)
    scratch_parent = REPO_ROOT / "flavourbench/artifacts/test-scratch"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_parent) as temporary:
        root = Path(temporary)
        coordinator = root / "coordinator"
        endpoints = {key: root / key for key in live_endpoints}
        coordinator.mkdir(parents=True)
        coordinator_line = (live_coordinator / "ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        (coordinator / "ledger.jsonl").write_text(
            coordinator_line + "\n", encoding="utf-8"
        )
        attestation_source = recovery._attestation_path(plan, REPO_ROOT)
        attestation_target = coordinator / "endpoint-attestations" / attestation_source.name
        attestation_target.parent.mkdir(parents=True)
        shutil.copyfile(attestation_source, attestation_target)
        for endpoint_id, endpoint_root in endpoints.items():
            endpoint_root.mkdir(parents=True)
            if endpoint_id == "gemini":
                first = (live_endpoints[endpoint_id] / "ledger.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
                (endpoint_root / "ledger.jsonl").write_text(first + "\n", encoding="utf-8")

        monkeypatch.setattr(executor, "_roots", lambda *_: (coordinator, endpoints))
        global_ledger = REPO_ROOT / "flavourbench/artifacts/frontier-contract/ledger.jsonl"
        monkeypatch.setattr(recovery, "GLOBAL_LEDGER_SHA256", study._file_sha256(global_ledger))
        monkeypatch.setattr(
            executor,
            "_attest_all_endpoints",
            lambda **_: (_ for _ in ()).throw(AssertionError("network path reached")),
        )
        packet = recovery.freeze(
            plan_path=PLAN,
            human_protocol_path=HUMAN,
            bound_preflight_path=BOUND,
            repo_root=REPO_ROOT,
            output_dir=root / "packet",
        )
        arguments = {
            "plan_path": PLAN,
            "human_protocol_path": HUMAN,
            "bound_preflight_path": BOUND,
            "contract_path": packet["contract"],
            "incident_path": packet["incident"],
            "repo_root": REPO_ROOT,
            "output_dir": root / "receipts",
            "confirmation": recovery.CONFIRMATION,
        }
        first_receipt = recovery.recover(**arguments)
        first_hashes = {
            path: study._file_sha256(path)
            for path in [
                coordinator / "ledger.jsonl",
                *(endpoints[key] / "ledger.jsonl" for key in sorted(endpoints)),
            ]
        }
        second_receipt = recovery.recover(**arguments)
        second_hashes = {path: study._file_sha256(path) for path in first_hashes}
        assert first_receipt == second_receipt
        assert first_hashes == second_hashes
        receipt = _document(first_receipt)
        assert receipt["coordinator_ledger"]["entries"] == 30
        assert {
            key: value["entries"] for key, value in receipt["endpoint_ledgers"].items()
        } == {"deepseek": 16, "gemini": 24, "sonnet": 16}
