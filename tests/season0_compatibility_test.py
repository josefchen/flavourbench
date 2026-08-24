from __future__ import annotations

from flavourbench.season0_compatibility import _normalize_final


def test_normalize_final_preserves_plain_text_without_format_gating() -> None:
    assert _normalize_final("Use toasted walnut as a bridge.") == {
        "answer_markdown": "Use toasted walnut as a bridge.",
        "ingredient_mentions": [],
        "constraints_addressed": [],
        "uncertainties": [],
    }
