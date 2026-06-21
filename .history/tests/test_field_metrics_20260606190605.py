from __future__ import annotations

from pathlib import Path

from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.report import evaluate_predictions


def make_pred(doc_id: str, header: dict, rows: list[dict[str, str]]) -> OCRPrediction:
    return OCRPrediction.model_validate({
        "document_id": doc_id,
        "method": "m",
        "header": header,
        "rows": rows,
    })


def write_gt(path: Path, rows: list[dict[str, str]]):
    import csv
    from ktt_ocr_eval.core.columns import ALL_COLUMNS

    keys = ALL_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in keys}
            writer.writerow(full)


def test_field_accuracy_perfect(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [{"lastname": "A", "firstname": "X", "age": "1"}]
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path)
    doc = report["documents"][0]
    fm = doc["field_metrics"]
    assert fm["lastname"]["correct"] == 1
    assert fm["lastname"]["accuracy"] == 1.0
    assert fm["firstname"]["correct"] == 1
    assert fm["firstname"]["accuracy"] == 1.0


def test_field_accuracy_mismatch(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [{"lastname": "Z", "firstname": "X", "age": "1"}]
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path)
    doc = report["documents"][0]
    fm = doc["field_metrics"]
    assert fm["lastname"]["correct"] == 0
    assert fm["lastname"]["accuracy"] == 0.0
    assert fm["firstname"]["correct"] == 1
    assert fm["firstname"]["accuracy"] == 1.0
