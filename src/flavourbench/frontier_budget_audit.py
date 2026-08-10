"""Conservatively audit spend across disjoint real frontier pilot strata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .frontier_contract_runner import scan_live_smoke_artifacts
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-frontier-global-budget-audit-v1"
DEFAULT_PRIOR_EXPOSURE_USD = Decimal("1.626135")


class FrontierBudgetAuditError(RuntimeError):
    """A source was duplicated or the global cost envelope was crossed."""


def audit_budget(
    source_directories: Sequence[Path],
    *,
    hard_cap_usd: Decimal = Decimal("100"),
    admission_fraction: Decimal = Decimal("0.85"),
    prior_exposure_usd: Decimal = DEFAULT_PRIOR_EXPOSURE_USD,
    next_reservation_usd: Decimal = Decimal(0),
) -> dict[str, Any]:
    if not source_directories:
        raise FrontierBudgetAuditError("at least one source directory is required")
    if hard_cap_usd <= 0 or not Decimal(0) < admission_fraction <= Decimal(1):
        raise FrontierBudgetAuditError("invalid global budget policy")
    if min(prior_exposure_usd, next_reservation_usd) < 0:
        raise FrontierBudgetAuditError("budget exposure cannot be negative")

    seen: set[str] = set()
    actual = Decimal(0)
    source_exposure = Decimal(0)
    by_provider: dict[str, dict[str, Decimal | int]] = {}
    by_basis: Counter[str] = Counter()
    inputs: list[dict[str, Any]] = []
    for source_directory in source_directories:
        scan = scan_live_smoke_artifacts(source_directory)
        directory_actual = Decimal(0)
        directory_exposure = Decimal(0)
        for item in scan.artifacts:
            if item.artifact_sha256 in seen:
                raise FrontierBudgetAuditError("duplicate source artifact across budget strata")
            seen.add(item.artifact_sha256)
            actual += item.actual_cost_usd
            source_exposure += item.exposure_usd
            directory_actual += item.actual_cost_usd
            directory_exposure += item.exposure_usd
            provider = item.requested_provider
            row = by_provider.setdefault(
                provider,
                {"source_count": 0, "actual_cost_usd": Decimal(0), "exposure_usd": Decimal(0)},
            )
            row["source_count"] = int(row["source_count"]) + 1
            row["actual_cost_usd"] = Decimal(row["actual_cost_usd"]) + item.actual_cost_usd
            row["exposure_usd"] = Decimal(row["exposure_usd"]) + item.exposure_usd
            by_basis[item.exposure_basis] += 1
        inputs.append(
            {
                "source_directory": str(source_directory),
                "source_count": len(scan.artifacts),
                "actual_cost_usd": str(directory_actual),
                "conservative_exposure_usd": str(directory_exposure),
                "artifact_set_sha256": sha256_json(
                    sorted(item.artifact_sha256 for item in scan.artifacts)
                ),
            }
        )

    current = prior_exposure_usd + source_exposure
    projected = current + next_reservation_usd
    admission_ceiling = hard_cap_usd * admission_fraction
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "currency": "USD",
        "hard_cap_usd": str(hard_cap_usd),
        "admission_fraction": str(admission_fraction),
        "admission_ceiling_usd": str(admission_ceiling),
        "prior_verified_exposure_usd": str(prior_exposure_usd),
        "real_source_count": len(seen),
        "real_source_actual_or_rate_card_estimate_usd": str(actual),
        "real_source_conservative_exposure_usd": str(source_exposure),
        "current_total_exposure_usd": str(current),
        "next_reservation_usd": str(next_reservation_usd),
        "projected_total_exposure_usd": str(projected),
        "admission_allowed": projected <= admission_ceiling,
        "hard_cap_respected": projected <= hard_cap_usd,
        "remaining_to_admission_ceiling_usd": str(admission_ceiling - projected),
        "remaining_to_hard_cap_usd": str(hard_cap_usd - projected),
        "inputs": inputs,
        "by_provider": {
            provider: {
                "source_count": int(row["source_count"]),
                "actual_cost_usd": str(row["actual_cost_usd"]),
                "conservative_exposure_usd": str(row["exposure_usd"]),
            }
            for provider, row in sorted(by_provider.items())
        },
        "exposure_basis_counts": dict(sorted(by_basis.items())),
        "synthetic_sources": 0,
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def write_audit(audit: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frontier-global-budget-{audit['artifact_sha256']}.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--hard-cap-usd", type=Decimal, default=Decimal("100"))
    parser.add_argument("--admission-fraction", type=Decimal, default=Decimal("0.85"))
    parser.add_argument(
        "--prior-exposure-usd", type=Decimal, default=DEFAULT_PRIOR_EXPOSURE_USD
    )
    parser.add_argument("--next-reservation-usd", type=Decimal, default=Decimal(0))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    audit = audit_budget(
        arguments.source_dir,
        hard_cap_usd=arguments.hard_cap_usd,
        admission_fraction=arguments.admission_fraction,
        prior_exposure_usd=arguments.prior_exposure_usd,
        next_reservation_usd=arguments.next_reservation_usd,
    )
    path = write_audit(audit, arguments.output_dir)
    print(
        json.dumps(
            {
                "status": "admissible" if audit["admission_allowed"] else "blocked",
                "artifact_sha256": audit["artifact_sha256"],
                "artifact": str(path.resolve()),
                "current_total_exposure_usd": audit["current_total_exposure_usd"],
                "projected_total_exposure_usd": audit["projected_total_exposure_usd"],
                "remaining_to_admission_ceiling_usd": audit[
                    "remaining_to_admission_ceiling_usd"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["admission_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
