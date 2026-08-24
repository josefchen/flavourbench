"""Run one unranked Epicure off/on pair through a frozen QwenCloud route."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

from .config import get_settings
from .direct_kimi_pair import _parser, _run_direct_pair
from .execution_policy import assert_legacy_paid_cli_allowed
from .qwencloud_smoke_admission import (
    CONDITIONS,
    EPICURE_MCP_URL,
    EPICURE_PROVENANCE_URL,
    EXECUTION_BACKEND,
    LIVE_CONFIRMATION,
    MODEL_ID,
    PROVIDER_SLUG,
    QwenCloudSmokeAdmissionError,
    begin_execution,
    record_execution_incident,
    terminalize_source,
    verify_go_template,
    verify_human_pi_authorization,
    verify_preflight_artifact,
)
from .service_qwencloud import (
    QWENCLOUD_ACCOUNTING_BASIS,
    QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS,
    QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS,
    QwenCloudDirectProvider,
)


async def run_pair(args):  # type: ignore[no-untyped-def]
    template = verify_go_template(
        args.go_template,
        expected_sha256=args.expected_go_template_sha256,
    )
    reservation = template["reservation"]
    model = template["model_identity"]
    task = template["task"]
    execution = template["execution"]
    settings = get_settings()
    provenance_url = os.environ.get("FLAVOURBENCH_EPICURE_PROVENANCE_URL")
    if not provenance_url:
        provenance_url = settings.mcp_url.removesuffix("/mcp").rstrip("/") + "/provenance"
    ledger_path = args.reservation_ledger.resolve()
    selected_conditions = tuple(args.condition or CONDITIONS)
    if (
        str(ledger_path) != reservation["ledger_path"]
        or args.reservation_entry_sha256 != reservation["entry_sha256"]
        or args.candidate_manifest_sha256 != model["route_manifest_sha256"]
        or args.model_id != MODEL_ID
        or args.provider_slug != PROVIDER_SLUG
        or args.expected_canonical_model_slug != MODEL_ID
        or args.expected_endpoint_execution_sha256 != model["endpoint_execution_sha256"]
        or args.expected_execution_policy_sha256 != execution["execution_policy_sha256"]
        or args.dataset_work_item_id != reservation["work_item_id"]
        or args.dataset_task_id != task["task_id"]
        or args.category != task["family"]
        or hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
        != task["prompt_sha256"]
        or Decimal(str(args.cap_usd)) != Decimal(str(reservation["full_ceiling_usd"]))
        or selected_conditions != CONDITIONS
        or args.plain_text_final is not True
        or args.evidence_protocol != "matched_evidence_v2"
        or args.require_epicure_call is not True
        or args.sequential_arms is not False
        or args.intermediate_reasoning_effort is not None
        or args.final_reasoning_effort is not None
        or args.allow_mutable_alias_exploratory is not True
        or args.tool_catalog_bytes_bound != 24_000
        or settings.mcp_url != EPICURE_MCP_URL
        or provenance_url != EPICURE_PROVENANCE_URL
        or execution.get("epicure_transport")
        != {
            "mcp_url": EPICURE_MCP_URL,
            "provenance_url": EPICURE_PROVENANCE_URL,
            "private_host_binding_required": True,
        }
    ):
        raise QwenCloudSmokeAdmissionError(
            "QwenCloud CLI arguments differ from the exact one-pair GO template"
        )
    args.frozen_run_id = execution["frozen_run_id"]
    args.frozen_attempt_slots = execution["attempt_slots"]

    async def execute() -> dict:  # type: ignore[type-arg]
        return await _run_direct_pair(
            args,
            execution_backend=EXECUTION_BACKEND,
            provider_factory=QwenCloudDirectProvider,
            credential_attribute="qwencloud_api_key",
            accounting_basis=QWENCLOUD_ACCOUNTING_BASIS,
            provider_label="QwenCloud",
            mutable_alias_accounting_basis=QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS,
            mutable_alias_billing_status=QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS,
        )

    if args.preflight_only:
        return await execute()

    if (
        args.preflight is None
        or not args.expected_preflight_sha256
        or args.human_pi_authorization is None
        or not args.expected_human_pi_authorization_sha256
    ):
        raise QwenCloudSmokeAdmissionError(
            "live QwenCloud execution requires the exact preflight and Human-PI GO"
        )
    preflight = verify_preflight_artifact(
        args.preflight,
        expected_sha256=args.expected_preflight_sha256,
        template=template,
    )
    authorization = verify_human_pi_authorization(
        args.human_pi_authorization,
        expected_sha256=args.expected_human_pi_authorization_sha256,
        template=template,
        preflight=preflight,
    )
    if (
        args.expected_epicure_release_id != preflight["epicure_release_id"]
        or args.expected_epicure_bundle_sha256 != preflight["epicure_bundle_sha256"]
        or args.expected_epicure_application_sha256
        != preflight["epicure_application_sha256"]
        or args.expected_epicure_tool_schema_sha256
        != preflight["epicure_tool_schema_sha256"]
    ):
        raise QwenCloudSmokeAdmissionError(
            "live Epicure expectation differs from the bound preflight"
        )
    if (
        settings.execution_mode != "live"
        or not settings.live_authorized
        or not settings.qwencloud_api_key
    ):
        raise QwenCloudSmokeAdmissionError(
            "live QwenCloud credential and explicit service authorization are required"
        )
    begin_execution(
        ledger_path=ledger_path,
        template=template,
        preflight=preflight,
        authorization=authorization,
        confirmation=args.live_confirm,
    )
    try:
        summary = await execute()
        artifact_path = Path(str(summary.get("artifact") or ""))
        if summary.get("status") != "complete_unpriced_budget_ceiling":
            raise QwenCloudSmokeAdmissionError(
                "QwenCloud pair did not produce a complete full-ceiling source"
            )
        terminal = terminalize_source(
            ledger_path=ledger_path,
            template=template,
            authorization=authorization,
            artifact_path=artifact_path,
        )
    except Exception as error:
        record_execution_incident(
            ledger_path=ledger_path,
            template=template,
            error=error,
        )
        raise
    return {
        **summary,
        "reservation_ledger": str(ledger_path),
        "reservation_entry_sha256": reservation["entry_sha256"],
        "execution_terminal_entry_sha256": terminal["entry_sha256"],
        "retained_exposure_usd": reservation["full_ceiling_usd"],
        "provider_cost_known": False,
    }


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-direct-qwencloud-pair")
    try:
        # QwenCloud does not document a low/high/max translation for this
        # frozen Chat Completions route.  Omission is part of the contract.
        parser = _parser(__doc__, reasoning_required=False)
        parser.add_argument(
            "--allow-mutable-alias-exploratory",
            action="store_true",
            help=(
                "Permit only a catalog-observed mutable alias contract; the result remains "
                "permanently exploratory, unranked, and unofficial."
            ),
        )
        parser.add_argument("--go-template", type=Path, required=True)
        parser.add_argument("--expected-go-template-sha256", required=True)
        parser.add_argument("--reservation-ledger", type=Path, required=True)
        parser.add_argument("--reservation-entry-sha256", required=True)
        parser.add_argument("--preflight", type=Path)
        parser.add_argument("--expected-preflight-sha256", default="")
        parser.add_argument("--human-pi-authorization", type=Path)
        parser.add_argument("--expected-human-pi-authorization-sha256", default="")
        parser.add_argument(
            "--live-confirm",
            default="",
            help=f"Required for live execution: {LIVE_CONFIRMATION}",
        )
        args = parser.parse_args()
        if args.frozen_attempt_slots_json:
            slots = json.loads(args.frozen_attempt_slots_json)
            if not isinstance(slots, list):
                raise RuntimeError("frozen attempt slots JSON must decode to an array")
            args.frozen_attempt_slots = slots
        summary = asyncio.run(run_pair(args))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] not in {
        "complete_rate_card_estimated",
        "complete_unpriced_budget_ceiling",
        "preflight_passed_no_provider_calls",
    }:
        sys.exit(2)


if __name__ == "__main__":
    run()
