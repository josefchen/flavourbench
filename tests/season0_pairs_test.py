from flavourbench.season0_pairs import _side_order, identity_leak_tags, round_robin_rounds


def test_twelve_model_round_robin_is_complete_and_balanced() -> None:
    models = [f"m{index:02d}" for index in range(12)]
    rounds = round_robin_rounds(models)
    assert len(rounds) == 11
    assert all(len(round_pairs) == 6 for round_pairs in rounds)
    assert all(
        sorted(model for pair in round_pairs for model in pair) == models
        for round_pairs in rounds
    )
    pairs = {tuple(sorted(pair)) for round_pairs in rounds for pair in round_pairs}
    assert len(pairs) == 66


def test_cryptographic_side_assignment_is_deterministic_and_balanced_over_seeds() -> None:
    assignments = [_side_order(f"seed-{index}", "left", "right") for index in range(1000)]
    assert assignments == [
        _side_order(f"seed-{index}", "left", "right") for index in range(1000)
    ]
    first_side_count = sum(value[0] == "left" for value in assignments)
    assert 450 <= first_side_count <= 550


def test_identity_leak_tags_detect_condition_route_and_model_names() -> None:
    assert identity_leak_tags("I queried Epicure MCP through Amazon Bedrock using Claude.") == [
        "epicure_condition",
        "provider_route",
        "model_name",
    ]
    assert identity_leak_tags("Use a bain-marie and whisk continuously.") == []
