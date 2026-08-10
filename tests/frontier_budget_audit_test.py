from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.frontier_budget_audit import FrontierBudgetAuditError, audit_budget

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/season1/current-quality-run/pilot-v27-eight-pairs/source"


def test_global_budget_audit_reproduces_conservative_real_pilot_exposure() -> None:
    audit = audit_budget([SOURCE])

    assert audit["real_source_count"] == 112
    assert Decimal(audit["real_source_actual_or_rate_card_estimate_usd"]) == Decimal(
        "9.468280"
    )
    assert Decimal(audit["real_source_conservative_exposure_usd"]) == Decimal(
        "17.32398673333333333333333333"
    )
    assert Decimal(audit["current_total_exposure_usd"]) == Decimal(
        "18.95012173333333333333333333"
    )
    assert audit["admission_allowed"] is True
    assert audit["synthetic_sources"] == 0


def test_global_budget_audit_rejects_duplicate_source_strata() -> None:
    with pytest.raises(FrontierBudgetAuditError, match="duplicate source artifact"):
        audit_budget([SOURCE, SOURCE])


def test_global_budget_audit_blocks_at_85_percent_before_hard_cap() -> None:
    audit = audit_budget([SOURCE], next_reservation_usd=Decimal("70"))

    assert audit["admission_allowed"] is False
    assert audit["hard_cap_respected"] is True
