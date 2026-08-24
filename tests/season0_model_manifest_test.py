from __future__ import annotations

from collections import Counter

import pytest

from flavourbench.season0_model_manifest import EXPECTED_ROLES, ModelManifestError


def test_expected_roles_define_exact_twelve_model_panel() -> None:
    assert sum(EXPECTED_ROLES.values()) == 12
    assert Counter(EXPECTED_ROLES) == Counter(
        {"closed_family": 4, "open_weight": 4, "efficiency": 2, "reasoning": 2}
    )


def test_model_manifest_error_is_explicit() -> None:
    with pytest.raises(ModelManifestError):
        raise ModelManifestError("missing real smoke")
