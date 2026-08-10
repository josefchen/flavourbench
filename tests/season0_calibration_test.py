from __future__ import annotations

from flavourbench.season0_calibration import _quantiles


def test_quantiles_are_deterministic_for_small_calibration_samples() -> None:
    assert _quantiles([]) == {"min": None, "median": None, "p95": None, "max": None}
    assert _quantiles([8, 1, 4, 2]) == {"min": 1, "median": 3, "p95": 8, "max": 8}
