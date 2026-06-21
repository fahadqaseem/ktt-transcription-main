from __future__ import annotations

from pathlib import Path

from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.report import evaluate_predictions


def make_pred(doc_id: str, method: str, header: dict, rows: list[dict[str, str]]) -> OCRPrediction:
    return OCRPrediction.model_validate({
        "document_id": doc_id,
        "method": method,
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


def test_method_field_accuracy_aggregation(tmp_path: Path):
    # ground truth has two records with distinct firstnames used as match keys
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Smith", "firstname": "Sam", "age": "10"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Jones", "firstname": "Bob", "age": "11"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}

    # two predictions for same method: one has lastname correct, one incorrect
    pred1 = make_pred("d1", "m", header, [{"firstname": "Sam", "lastname": "Smith", "age": "10"}])
    pred2 = make_pred("d2", "m", header, [{"firstname": "Bob", "lastname": "Wrong", "age": "11"}])

    report = evaluate_predictions([pred1, pred2], gt_path, config={"match_key_fields": ["firstname"]})
    methods = report["summary"]["methods"]
    assert "m" in methods
    fa = methods["m"]["field_accuracy"]
    # lastname correct in 1 of 2 matched rows
    assert fa.get("lastname") == 0.5
