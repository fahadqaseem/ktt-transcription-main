from __future__ import annotations

from pathlib import Path

from ktt_ocr_eval.orchestration.render import (
    build_table,
    fractional_ranks,
    load_scores,
    render_table,
)


def _row(method: str, metric: str, value: float, *, nickname: str | None = None) -> dict:
    return {
        "method": method,
        "fingerprint": method[:3],
        "nickname": nickname,
        "config": {},
        "metric": metric,
        "field": None,
        "value": value,
    }


def test_fractional_ranks_direction_and_ties() -> None:
    assert fractional_ranks([0.9, 0.8, 0.7], lower_is_better=False) == [1.0, 2.0, 3.0]
    assert fractional_ranks([0.7, 0.8, 0.9], lower_is_better=True) == [1.0, 2.0, 3.0]
    assert fractional_ranks([0.9, 0.9, 0.8], lower_is_better=False) == [1.5, 1.5, 3.0]


def test_order_uses_mean_rank_not_value() -> None:
    # A wins f1 but loses recall; B the reverse -> both mean-rank 1.5 (a tie).
    rows = [
        _row("A", "f1", 0.9),
        _row("A", "recall", 0.5),
        _row("B", "f1", 0.5),
        _row("B", "recall", 0.9),
    ]
    _, _, _, avg_rank = build_table(rows, ["f1", "recall"])
    assert avg_rank["A@A"] == 1.5
    assert avg_rank["B@B"] == 1.5


def test_failure_rate_ranks_lower_is_better() -> None:
    rows = [
        _row("A", "f1", 0.5),
        _row("A", "failure_rate", 0.0),
        _row("B", "f1", 0.5),
        _row("B", "failure_rate", 0.5),
    ]
    ordered, _, _, _ = build_table(rows, ["f1", "failure_rate"])
    assert ordered[0] == "A@A"  # ties on f1, but A has the lower failure rate


def test_render_table_lists_methods_best_first() -> None:
    rows = [_row("good", "f1", 0.9, nickname="good"), _row("bad", "f1", 0.1, nickname="bad")]
    lines = render_table(rows, ["f1"]).splitlines()
    assert lines[0].split()[0] == "method"
    body = lines[2:]
    assert body[0].startswith("good")
    assert body[1].startswith("bad")


def test_load_scores_reads_file_and_latest_in_dir(tmp_path: Path) -> None:
    older = tmp_path / "scores-20260101T000000-aaaaaa.jsonl"
    older.write_text('{"method":"m","metric":"f1","value":0.5}\n\n', encoding="utf-8")
    assert load_scores(older) == [{"method": "m", "metric": "f1", "value": 0.5}]
    # a folder resolves to the latest scores-*.jsonl
    newer = tmp_path / "scores-20260102T000000-bbbbbb.jsonl"
    newer.write_text('{"method":"n","metric":"f1","value":0.9}\n', encoding="utf-8")
    assert load_scores(tmp_path) == [{"method": "n", "metric": "f1", "value": 0.9}]
