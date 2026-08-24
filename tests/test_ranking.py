from __future__ import annotations

from flavourbench.ranking import _fit_bradley_terry, _paired_ordinal, _wilson


def test_bradley_terry_recovers_known_order_with_ties() -> None:
    comparisons = []
    comparisons.extend(("strong", "middle", 1.0) for _ in range(30))
    comparisons.extend(("middle", "weak", 1.0) for _ in range(24))
    comparisons.extend(("strong", "weak", 1.0) for _ in range(30))
    comparisons.extend(("strong", "middle", 0.5) for _ in range(6))
    ratings = _fit_bradley_terry(comparisons)
    assert ratings["strong"][0] > ratings["middle"][0] > ratings["weak"][0]
    assert all(low <= rating <= high for rating, low, high in ratings.values())
    strict_ratings = _fit_bradley_terry(comparisons, require_arena_rank=True)
    assert strict_ratings["strong"][0] > strict_ratings["middle"][0] > strict_ratings["weak"][0]


def test_disconnected_or_null_graph_is_handled() -> None:
    assert _fit_bradley_terry([]) == {}
    ratings = _fit_bradley_terry([("a", "b", 0.5) for _ in range(20)])
    assert abs(ratings["a"][0] - ratings["b"][0]) < 1


def test_uplift_interval_contains_null_for_balanced_effect() -> None:
    low, high = _wilson(25, 50)
    assert low < 0.5 < high
    estimate, ordinal_low, ordinal_high = _paired_ordinal(20, 10, 20)
    assert abs(estimate - 0.5) < 0.02
    assert ordinal_low < 0.5 < ordinal_high


def test_paired_ordinal_recovers_epicure_effect_with_ties() -> None:
    estimate, low, high = _paired_ordinal(40, 8, 12)
    assert estimate > 0.6
    assert low < estimate < high
