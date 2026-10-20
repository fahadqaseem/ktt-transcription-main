from __future__ import annotations

import json
from pathlib import Path

from ktt_ocr_eval.core.artifacts import write_predictions
from ktt_ocr_eval.core.schemas import CollectionInfo, OCRPrediction
from ktt_ocr_eval.orchestration.scoring import (
    group_by_variant,
    is_failure,
    load_predictions_tree,
    score_suite,
    variant_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth.example.csv"


def _pred(
    document_id: str,
    method: str,
    fingerprint: str,
    *,
    nickname: str | None = None,
    error: str | None = None,
) -> OCRPrediction:
    return OCRPrediction(
        document_id=document_id,
        method=method,
        collection=CollectionInfo(
            fingerprint=fingerprint, adapter="x", nickname=nickname, config={"k": 1}
        ),
        metadata={"error": error, "error_type": "Boom"} if error else {},
    )


def test_is_failure_detects_error_records() -> None:
    assert is_failure(_pred("d", "m", "f", error="boom"))
    assert not is_failure(_pred("d", "m", "f"))


def test_group_by_variant_splits_by_method_and_fingerprint() -> None:
    predictions = [
        _pred("d1", "m", "aaa"),
        _pred("d2", "m", "aaa"),
        _pred("d1", "m", "bbb"),  # same method, different config -> separate variant
        _pred("d1", "n", "aaa"),  # different method
    ]
    groups = group_by_variant(predictions)
    assert set(groups) == {("m", "aaa"), ("m", "bbb"), ("n", "aaa")}
    assert len(groups[("m", "aaa")]) == 2


def test_variant_label_prefers_nickname_else_method_and_fingerprint() -> None:
    assert variant_label(_pred("d", "m", "abcdef123456", nickname="Tuned GPT")) == "Tuned GPT"
    assert variant_label(_pred("d", "m", "abcdef123456")) == "m@abcdef12"


def test_load_predictions_tree_globs_directory(tmp_path: Path) -> None:
    write_predictions(tmp_path / "predictions-1.jsonl", [_pred("d1", "m", "aaa")])
    write_predictions(tmp_path / "predictions-2.jsonl", [_pred("d2", "m", "aaa")])
    assert len(load_predictions_tree(tmp_path)) == 2


def test_score_suite_emits_tidy_metric_rows_with_failures(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_predictions(
        raw / "predictions-1.jsonl",
        [
            _pred("d1", "m", "aaa", nickname="A"),
            _pred("d2", "m", "aaa", nickname="A"),
            _pred("d1", "m", "bbb", nickname="B"),
            _pred("d2", "m", "bbb", nickname="B", error="rate limit"),  # one failure
        ],
    )
    out = tmp_path / "scores"
    scores_path, rows = score_suite(raw, GROUND_TRUTH, out, {})

    # tidy long format: every row is self-describing with method/config + metric/value
    required = {"method", "fingerprint", "nickname", "config", "metric", "field", "value"}
    assert all(required <= set(row) for row in rows)
    assert any(row["config"] == {"k": 1} for row in rows)

    # one tidy jsonl per run; no fixed scores.jsonl, no per-variant dirs in this stage
    assert scores_path.exists()
    assert scores_path.name.startswith("scores-") and scores_path.suffix == ".jsonl"
    assert not (out / "scores.jsonl").exists()
    assert not (out / "A").exists()

    def values(fingerprint: str, metric: str) -> list[object]:
        return [r["value"] for r in rows if r["fingerprint"] == fingerprint and r["metric"] == metric]

    assert values("aaa", "failure_rate") == [0.0]
    assert values("bbb", "failure_rate") == [0.5]
    assert values("bbb", "records_failed") == [1]
    assert {r["fingerprint"] for r in rows if r["metric"] == "f1"} == {"aaa", "bbb"}

    field_rows = [r for r in rows if r["metric"] == "field_accuracy"]
    assert field_rows and all(r["field"] for r in field_rows)

    on_disk = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert len(on_disk) == len(rows)
