from flavourbench.season0_cost_envelope import SCHEMA_VERSION


def test_cost_envelope_schema_is_versioned() -> None:
    assert SCHEMA_VERSION == "flavourbench-season0-cost-envelope-v1"
