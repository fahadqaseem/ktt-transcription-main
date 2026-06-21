from __future__ import annotations

from ktt_ocr_eval.evaluation.comparison import match_by_exact_key


def make_row(lastname: str, firstname: str, age: str = "") -> dict:
    return {"lastname": lastname, "firstname": firstname, "age": age}


def test_perfect_match():
    expected = [make_row("Doe", "John", "10")]
    predicted = [make_row("Doe", "John", "10")]

    out = match_by_exact_key(predicted, expected)
    assert out["matches"] == [(0, 0)]
    assert out["pred_unmatched"] == []
    assert out["exp_unmatched"] == []


def test_extra_predicted_is_fp():
    expected = [make_row("Doe", "John", "10")]
    predicted = [make_row("Doe", "John", "10"), make_row("Smith", "Ann", "9")]

    out = match_by_exact_key(predicted, expected)
    assert out["matches"] == [(0, 0)]
    assert out["pred_unmatched"] == [1]
    assert out["exp_unmatched"] == []


def test_missing_predicted_is_fn():
    expected = [make_row("Doe", "John", "10"), make_row("Smith", "Ann", "9")]
    predicted = [make_row("Doe", "John", "10")]

    out = match_by_exact_key(predicted, expected)
    assert out["matches"] == [(0, 0)]
    assert out["pred_unmatched"] == []
    assert sorted(out["exp_unmatched"]) == [1]


def test_duplicate_names_match_in_order():
    expected = [make_row("Lee", "Sam"), make_row("Lee", "Sam")]
    predicted = [make_row("Lee", "Sam"), make_row("Lee", "Sam")]

    out = match_by_exact_key(predicted, expected)
    assert out["matches"] == [(0, 0), (1, 1)]
    assert out["pred_unmatched"] == []
    assert out["exp_unmatched"] == []
