"""Finite direct Z.ai GLM-5.3 adapter for the powered benchmark CLI.

The ordinary GLM Coding Plan terms limit the endpoint to supported tools.
Z.ai separately gave Josef Chen written permission for one limited benchmark
run, provided it is not exposed as a permanent function.  The frozen backend
contract carries that scope; this adapter rejects contracts that omit it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .provider import GenerationFailureResult, GenerationResult, GenerationSpec, ProviderError
from .service_kimi import KimiDirectProvider

ZAI_CODING_CONTRACT_SCHEMA = "flavourbench-zai-coding-anthropic-contract-v1"
ZAI_CODING_PROVIDER_SLUG = "zai-coding-plan-direct"
ZAI_CODING_ACCOUNTING_BASIS = "zai_coding_plan_subscription_quota"
ZAI_CODING_BILLING_STATUS = "subscription_quota_consumed_provider_charge_unavailable"


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


class ZaiCodingDirectProvider(KimiDirectProvider):
    """Execute the one authorized GLM-5.3 block without a standing service."""

    direct_provider_name = "Z.ai GLM Coding Plan"
    execution_backend = "zai_coding_direct"
    provider_slug = ZAI_CODING_PROVIDER_SLUG
    contract_schema = ZAI_CODING_CONTRACT_SCHEMA
    credential_setting = "zai_coding_api_key"
    credential_environment_name = "FLAVOURBENCH_ZAI_CODING_API_KEY"
    base_url_setting = "zai_coding_base_url"
    timeout_setting = "zai_coding_timeout_seconds"
    accounting_basis = ZAI_CODING_ACCOUNTING_BASIS

    def _authorization_headers(self, api_key: str) -> dict[str, str]:
        # Z.ai documents ANTHROPIC_AUTH_TOKEN for this compatibility endpoint.
        return {"Authorization": f"Bearer {api_key}"}

    def _validate_contract(self, spec: GenerationSpec) -> str:
        requested_model = super()._validate_contract(spec)
        contract = spec.backend_contract_json
        permission = contract.get("limited_run_permission")
        if not isinstance(permission, Mapping):
            raise ProviderError("Z.ai execution lacks its frozen limited-run permission")
        if (
            requested_model != "glm-5.3"
            or contract.get("identity_kind") != "official_named_release"
            or contract.get("rank_eligible_after_complete_block") is not True
            or permission.get("provider") != "Z.ai"
            or permission.get("grantee") != "Josef Chen"
            or permission.get("scope") != "one_finite_flavourbench_benchmark_run"
            or permission.get("permanent_running_function_authorized") is not False
            or permission.get("user_attested_written_permission") is not True
            or not _is_sha256(permission.get("permission_quote_sha256"))
        ):
            raise ProviderError("Z.ai limited-run permission differs from the frozen contract")
        return requested_model

    def _rate_card_accounting(
        self,
        *,
        generation_id: str,
        arm_id: str,
        response_model: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        accounting = super()._rate_card_accounting(
            generation_id=generation_id,
            arm_id=arm_id,
            response_model=response_model,
            usage=usage,
        )
        accounting.update(
            {
                "cost_micros": 0,
                "reconciled": False,
                "accounting_basis": self.accounting_basis,
                "billing_reconciliation_status": ZAI_CODING_BILLING_STATUS,
            }
        )
        return accounting

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        result = await super().generate(spec)
        result.cost_micros = 0
        result.cost_reconciled = False
        result.cost_accounting_basis = self.accounting_basis
        result.billing_reconciliation_status = ZAI_CODING_BILLING_STATUS
        return result

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        result = await super().reconcile_failure(spec, error)
        if result is not None:
            result.cost_micros = 0
            result.cost_reconciled = False
            result.cost_accounting_basis = self.accounting_basis
            result.billing_reconciliation_status = ZAI_CODING_BILLING_STATUS
        return result
