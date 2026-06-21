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


def test_prf_perfect(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "B", "firstname": "Y", "age": "2"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "A", "firstname": "X", "age": "1"},
        {"lastname": "B", "firstname": "Y", "age": "2"},
    ]
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path)
    summary = report["summary"]
    assert summary["tp"] == 2
    assert summary["fp"] == 0
    assert summary["fn"] == 0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["f1"] == 1.0


def test_prf_extra_predicted(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "A", "firstname": "X", "age": "1"},
        {"lastname": "Z", "firstname": "Q", "age": "9"},
    ]
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path)
    summary = report["summary"]
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 1.0
    assert round(summary["f1"], 4) == 0.6667


def test_prf_missing_predicted(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "B", "firstname": "Y", "age": "2"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "A", "firstname": "X", "age": "1"},
    ]
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path)
    summary = report["summary"]
    assert summary["tp"] == 1
    assert summary["fp"] == 0
    assert summary["fn"] == 1
    assert summary["precision"] == 1.0
    assert summary["recall"] == 0.5
    assert round(summary["f1"], 4) == 0.6667
