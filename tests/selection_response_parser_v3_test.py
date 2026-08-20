from __future__ import annotations

from flavourbench.selection_response_parser_v3 import parse_final_selection_v3


def _task() -> dict[str, object]:
    return {
        "choices": {
            "A": "basil",
            "B": "black_pepper",
            "C": "quinoa",
            "D": "lemon_balm",
            "E": "red_currant",
            "F": "cherry",
            "G": "grenadine",
            "H": "mint",
        }
    }


def test_v3_accepts_exact_labels_names_and_known_provider_close_token() -> None:
    task = _task()
    assert parse_final_selection_v3(task, "FINAL_SELECTION: C,A,B") == "ABC"
    assert parse_final_selection_v3(task, "FINAL_SELECTION: basil, black pepper, quinoa") == "ABC"
    assert parse_final_selection_v3(task, "FINAL_SELECTION:B,D,G<|close|>response") == "BDG"


def test_v3_accepts_one_unambiguous_corrected_selection() -> None:
    answer = (
        "FINAL_SELECTION: G, H, lard is not valid; the correct response is: "
        "FINAL_SELECTION: A, G, H"
    )
    assert parse_final_selection_v3(_task(), answer) == "AGH"


def test_v3_fails_closed_on_ambiguity_or_invalid_cardinality() -> None:
    task = _task()
    assert parse_final_selection_v3(task, "FINAL_SELECTION: A,B,C\nFINAL_SELECTION: A,B,D") is None
    assert parse_final_selection_v3(task, "FINAL_SELECTION: A,A,B") is None
    assert parse_final_selection_v3(task, "FINAL_SELECTION: A,B,Z") is None
    assert parse_final_selection_v3(task, "FINAL_SELECTION: A,B,C,D") is None
    assert parse_final_selection_v3(task, "FINAL_SELECTION: A,H") is None


def test_v3_requires_the_marker_and_exact_candidate_names() -> None:
    task = _task()
    assert parse_final_selection_v3(task, "basil, black pepper, quinoa") is None
    assert parse_final_selection_v3(task, "FINAL_SELECTION: basil, pepper, quinoa") is None
