from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "flavourbench-season1-statistical-method-validation-v1"
MONTE_CARLO_DATASETS = 2_000
BOOTSTRAP_REPLICATES = 999
VALIDATION_SEED = 20260801
FAMILIES = ("composition", "cookability", "evidence", "substitution")
TASKS_PER_FAMILY = 50
NULL_VALUE = 0.5
MIN_COVERAGE = 0.93
MAX_COVERAGE = 0.97
MIN_TYPE_I_ERROR = 0.04
MAX_TYPE_I_ERROR = 0.06
MIN_POWER = 0.80

def _project_root() -> Path:
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "requirements.lock").is_file():
            return candidate
    return candidates[-1]


ROOT = _project_root()
DEFAULT_OUTPUT_DIR = ROOT / "contracts/season1/method-validation"
STATISTICS_SOURCE = Path(__file__).with_name("season1_statistics.py")
REQUIREMENTS_LOCK = ROOT / "requirements.lock"


class MethodValidationError(ValueError):
    """The statistical method-validation contract is malformed or fails."""


@dataclass(frozen=True)
class OrdinalScenario:
    scenario_id: str
    loss_probability: float
    tie_probability: float
    win_probability: float
    alternative: bool

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (
            self.loss_probability,
            self.tie_probability,
            self.win_probability,
        )

    @property
    def true_half_win_share(self) -> float:
        return self.win_probability + 0.5 * self.tie_probability

    def validate(self) -> None:
        if not self.scenario_id:
            raise MethodValidationError("scenario id is required")
        if any(value < 0 or value > 1 for value in self.probabilities):
            raise MethodValidationError("ordinal probabilities must lie in [0, 1]")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-12):
            raise MethodValidationError("ordinal probabilities must sum to one")
        if self.alternative == math.isclose(
            self.true_half_win_share, NULL_VALUE, abs_tol=1e-12
        ):
            raise MethodValidationError("scenario alternative flag contradicts its estimand")


SCENARIOS = (
    OrdinalScenario(
        scenario_id="symmetric-null-with-ties",
        loss_probability=0.40,
        tie_probability=0.20,
        win_probability=0.40,
        alternative=False,
    ),
    OrdinalScenario(
        scenario_id="positive-practical-effect-with-ties",
        loss_probability=0.30,
        tie_probability=0.20,
        win_probability=0.50,
        alternative=True,
    ),
)

_REPRODUCTION_CACHE: dict[str, bool] = {}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rating_difference(half_win_share: float) -> float:
    clipped = min(max(half_win_share, 1e-12), 1.0 - 1e-12)
    return (400.0 / math.log(10.0)) * math.log(clipped / (1.0 - clipped))


def _mcse(rate: float, datasets: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / datasets)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _family_stratified_interval(
    family_outcomes: np.ndarray,
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Mirror the prospective equal-family, task-cluster percentile bootstrap.

    Each simulated row is one independent task cluster. Production observations
    may contain repeated raters and generations within a task; their hierarchical
    weights collapse to this exact primitive in the one-row-per-task subproblem.
    """

    family_count, tasks_per_family = family_outcomes.shape
    bootstrap = np.zeros(bootstrap_replicates, dtype=float)
    for family_index in range(family_count):
        sampled_indices = rng.integers(
            0,
            tasks_per_family,
            size=(bootstrap_replicates, tasks_per_family),
        )
        bootstrap += (
            family_outcomes[family_index][sampled_indices].mean(axis=1) / family_count
        )
    low, high = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    return float(low), float(high)


def evaluate_scenario(
    scenario: OrdinalScenario,
    *,
    datasets: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    scenario.validate()
    if datasets < 1 or bootstrap_replicates < 1:
        raise MethodValidationError("dataset and bootstrap counts must be positive")

    rng = np.random.default_rng(seed)
    true_share = scenario.true_half_win_share
    estimates = np.empty(datasets, dtype=float)
    lower = np.empty(datasets, dtype=float)
    upper = np.empty(datasets, dtype=float)
    values = np.asarray((0.0, 0.5, 1.0), dtype=float)

    for dataset_index in range(datasets):
        outcomes = rng.choice(
            values,
            size=(len(FAMILIES), TASKS_PER_FAMILY),
            p=scenario.probabilities,
        )
        estimates[dataset_index] = float(outcomes.mean())
        lower[dataset_index], upper[dataset_index] = _family_stratified_interval(
            outcomes,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )

    covered = (lower <= true_share) & (true_share <= upper)
    rejects_null = (upper < NULL_VALUE) | (lower > NULL_VALUE)
    positive_rejects = lower > NULL_VALUE
    coverage = float(covered.mean())
    type_i_error = float(rejects_null.mean()) if not scenario.alternative else None
    power = float(positive_rejects.mean()) if scenario.alternative else None
    coverage_pass = MIN_COVERAGE <= coverage <= MAX_COVERAGE
    type_i_pass = (
        type_i_error is None
        or MIN_TYPE_I_ERROR <= type_i_error <= MAX_TYPE_I_ERROR
    )
    power_pass = power is None or power >= MIN_POWER
    uplift = {
        "estimand": "family-standardized Epicure half-win share",
        "true_value": _rounded(true_share),
        "mean_estimate": _rounded(float(estimates.mean())),
        "bias": _rounded(float(estimates.mean() - true_share)),
        "interval_coverage": _rounded(coverage),
        "coverage_mcse": _rounded(_mcse(coverage, datasets)),
        "two_sided_type_i_error": (
            _rounded(type_i_error) if type_i_error is not None else None
        ),
        "type_i_error_mcse": (
            _rounded(_mcse(type_i_error, datasets)) if type_i_error is not None else None
        ),
        "one_sided_positive_power": _rounded(power) if power is not None else None,
        "power_mcse": _rounded(_mcse(power, datasets)) if power is not None else None,
    }

    true_rating = _rating_difference(true_share)
    rating_estimates = np.asarray([_rating_difference(value) for value in estimates])
    rating_lower = np.asarray([_rating_difference(value) for value in lower])
    rating_upper = np.asarray([_rating_difference(value) for value in upper])
    rating_covered = (rating_lower <= true_rating) & (true_rating <= rating_upper)
    arena = {
        "estimand": "two-endpoint Bradley-Terry rating difference",
        "true_value": _rounded(true_rating),
        "mean_estimate": _rounded(float(rating_estimates.mean())),
        "bias": _rounded(float(rating_estimates.mean() - true_rating)),
        "interval_coverage": _rounded(float(rating_covered.mean())),
        "two_sided_type_i_error": (
            _rounded(type_i_error) if type_i_error is not None else None
        ),
        "one_sided_positive_power": _rounded(power) if power is not None else None,
        "scope": "identifiable two-endpoint subproblem of the frozen Bradley-Terry estimator",
    }
    status = "pass" if coverage_pass and type_i_pass and power_pass else "fail"
    return {
        "scenario_id": scenario.scenario_id,
        "ordinal_probabilities": {
            "loss": scenario.loss_probability,
            "tie": scenario.tie_probability,
            "win": scenario.win_probability,
        },
        "monte_carlo_datasets": datasets,
        "tasks_per_family": TASKS_PER_FAMILY,
        "families": list(FAMILIES),
        "bootstrap_replicates_per_dataset": bootstrap_replicates,
        "bootstrap_method": "family-stratified task-cluster percentile bootstrap",
        "uplift": uplift,
        "model_arena": arena,
        "acceptance": {
            "status": status,
            "coverage_in_0.93_0.97": coverage_pass,
            "null_type_i_in_0.04_0.06": type_i_pass,
            "alternative_power_at_least_0.80": power_pass,
        },
    }


def build_artifact(
    *,
    datasets: int = MONTE_CARLO_DATASETS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = VALIDATION_SEED,
) -> dict[str, Any]:
    scenario_rows = [
        evaluate_scenario(
            scenario,
            datasets=datasets,
            bootstrap_replicates=bootstrap_replicates,
            seed=int(np.random.SeedSequence((seed, index)).generate_state(1)[0]),
        )
        for index, scenario in enumerate(SCENARIOS)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_class": "simulated_statistical_method_validation_only",
        "protocol_date": "2026-08-01",
        "parameters": {
            "monte_carlo_datasets_per_scenario": datasets,
            "bootstrap_replicates_per_dataset": bootstrap_replicates,
            "seed": seed,
            "families": list(FAMILIES),
            "tasks_per_family": TASKS_PER_FAMILY,
            "tie_treatment": "half-win",
        },
        "acceptance_thresholds": {
            "interval_coverage": [MIN_COVERAGE, MAX_COVERAGE],
            "two_sided_type_i_error": [MIN_TYPE_I_ERROR, MAX_TYPE_I_ERROR],
            "one_sided_positive_power_minimum": MIN_POWER,
        },
        "scenarios": scenario_rows,
        "source_hashes": {
            "method_validation_py": sha256_file(Path(__file__)),
            "season1_statistics_py": sha256_file(STATISTICS_SOURCE),
            "requirements_lock": sha256_file(REQUIREMENTS_LOCK),
        },
        "claim_boundary": {
            "scored_benchmark_observations": 0,
            "leaderboard_use": False,
            "model_quality_claim": False,
            "epicure_quality_claim": False,
            "purpose": "finite-sample validation of the pre-registered estimators",
            "limitation": (
                "This validates a balanced identifiable subproblem. It does not establish "
                "construct validity, rater validity, contamination control, or empirical model "
                "performance. Those require the frozen real Season 1 collection."
            ),
        },
        "acceptance": {
            "status": (
                "pass"
                if datasets >= MONTE_CARLO_DATASETS
                and all(row["acceptance"]["status"] == "pass" for row in scenario_rows)
                else "fail"
            ),
            "minimum_monte_carlo_datasets_per_scenario": MONTE_CARLO_DATASETS,
            "all_scenarios_pass": all(
                row["acceptance"]["status"] == "pass" for row in scenario_rows
            ),
        },
    }
    return {**payload, "artifact_sha256": canonical_sha256(payload)}


def verify_artifact(value: dict[str, Any], *, reproduce: bool = False) -> bool:
    artifact_sha256 = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    parameters = value.get("parameters")
    acceptance = value.get("acceptance")
    claim_boundary = value.get("claim_boundary")
    scenarios = value.get("scenarios")
    source_hashes = value.get("source_hashes")
    structurally_valid = bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_class") == "simulated_statistical_method_validation_only"
        and artifact_sha256 == canonical_sha256(payload)
        and isinstance(parameters, dict)
        and int(parameters.get("monte_carlo_datasets_per_scenario", 0))
        >= MONTE_CARLO_DATASETS
        and int(parameters.get("bootstrap_replicates_per_dataset", 0)) >= 1
        and isinstance(scenarios, list)
        and len(scenarios) == len(SCENARIOS)
        and all(
            isinstance(row, dict)
            and int(row.get("monte_carlo_datasets", 0)) >= MONTE_CARLO_DATASETS
            and row.get("acceptance", {}).get("status") == "pass"
            for row in scenarios
        )
        and isinstance(acceptance, dict)
        and acceptance.get("status") == "pass"
        and acceptance.get("all_scenarios_pass") is True
        and isinstance(claim_boundary, dict)
        and claim_boundary.get("scored_benchmark_observations") == 0
        and claim_boundary.get("leaderboard_use") is False
        and isinstance(source_hashes, dict)
        and source_hashes.get("method_validation_py") == sha256_file(Path(__file__))
        and source_hashes.get("season1_statistics_py") == sha256_file(STATISTICS_SOURCE)
        and source_hashes.get("requirements_lock") == sha256_file(REQUIREMENTS_LOCK)
    )
    if not structurally_valid or not reproduce:
        return structurally_valid
    cached = _REPRODUCTION_CACHE.get(str(artifact_sha256))
    if cached is not None:
        return cached
    expected = build_artifact(
        datasets=int(parameters["monte_carlo_datasets_per_scenario"]),
        bootstrap_replicates=int(parameters["bootstrap_replicates_per_dataset"]),
        seed=int(parameters["seed"]),
    )
    reproduced = canonical_bytes(value) == canonical_bytes(expected)
    _REPRODUCTION_CACHE[str(artifact_sha256)] = reproduced
    return reproduced


def write_artifact(value: dict[str, Any], output_dir: Path) -> Path:
    if not verify_artifact(value, reproduce=True):
        raise MethodValidationError("refusing to publish a failed method-validation artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"season1-statistical-method-validation-{value['artifact_sha256']}.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=int, default=MONTE_CARLO_DATASETS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    artifact = build_artifact(
        datasets=args.datasets,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(write_artifact(artifact, args.output_dir.resolve()))


if __name__ == "__main__":
    run()
