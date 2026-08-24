"""Pure admission semantics for the isolated Bedrock spend lane.

Persistence and row locking belong to the eventual worker integration.  This
module deliberately accepts a transactionally-read snapshot and returns a
decision that a PostgreSQL ledger can commit atomically with its reservation.
It performs no I/O and cannot spend money.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .bedrock_auth import BedrockLaneSettings

AdmissionStatus = Literal["admit", "stop_admission", "drain_only", "hard_stop"]


@dataclass(frozen=True)
class BedrockBudgetSnapshot:
    actual_spend_usd: Decimal
    outstanding_reservations_usd: Decimal

    def __post_init__(self) -> None:
        if self.actual_spend_usd < 0 or self.outstanding_reservations_usd < 0:
            raise ValueError("Bedrock budget values must be non-negative")

    @property
    def exposure_usd(self) -> Decimal:
        return self.actual_spend_usd + self.outstanding_reservations_usd


@dataclass(frozen=True)
class BedrockAdmissionDecision:
    status: AdmissionStatus
    admitted: bool
    stage: str
    requested_reservation_usd: Decimal
    exposure_before_usd: Decimal
    exposure_after_usd: Decimal
    effective_stage_cap_usd: Decimal
    hard_cap_usd: Decimal
    reason: str


class BedrockCostGovernor:
    """Apply 85% admission, 95% drain, and 100% hard-stop boundaries."""

    admission_fraction = Decimal("0.85")
    drain_fraction = Decimal("0.95")

    def __init__(self, settings: BedrockLaneSettings) -> None:
        self.settings = settings

    def decide(
        self,
        snapshot: BedrockBudgetSnapshot,
        *,
        worst_case_reservation_usd: Decimal,
    ) -> BedrockAdmissionDecision:
        if worst_case_reservation_usd <= 0:
            raise ValueError("worst-case Bedrock reservation must be positive")

        stage_cap = self.settings.effective_stage_cap_usd
        hard_cap = self.settings.hard_cap_usd
        before = snapshot.exposure_usd
        after = before + worst_case_reservation_usd

        def decision(status: AdmissionStatus, reason: str) -> BedrockAdmissionDecision:
            return BedrockAdmissionDecision(
                status=status,
                admitted=status == "admit",
                stage=self.settings.stage,
                requested_reservation_usd=worst_case_reservation_usd,
                exposure_before_usd=before,
                exposure_after_usd=after,
                effective_stage_cap_usd=stage_cap,
                hard_cap_usd=hard_cap,
                reason=reason,
            )

        if not self.settings.enabled or not self.settings.live_authorized:
            return decision("hard_stop", "Bedrock live execution is not explicitly authorized")
        if hard_cap <= 0 or before >= hard_cap or after > hard_cap:
            return decision("hard_stop", "the authorized Bedrock hard cap would be exceeded")
        if stage_cap <= 0 or before >= stage_cap or after > stage_cap:
            return decision("hard_stop", "the active Bedrock stage cap would be exceeded")
        if before >= stage_cap * self.drain_fraction:
            return decision("drain_only", "the active stage is at its 95% drain boundary")
        if before >= stage_cap * self.admission_fraction:
            return decision("stop_admission", "the active stage is at its 85% admission boundary")
        if after > stage_cap * self.admission_fraction:
            return decision(
                "stop_admission",
                "the reservation would cross the active stage's 85% admission boundary",
            )
        return decision("admit", "worst-case reservation is within every active boundary")
