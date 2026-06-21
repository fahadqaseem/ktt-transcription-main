from __future__ import annotations

from pathlib import Path

from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.report import evaluate_predictions


def write_gt(path: Path, rows: list[dict[str, str]]):
    # write a minimal CSV with header+rows using keys from rows[0]
    import csv

    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_pred(doc_id: str, header: dict, rows: list[dict[str, str]]) -> OCRPrediction:
    return OCRPrediction.model_validate({
        "document_id": doc_id,
        "method": "test-method",
        "header": header,
        "rows": rows,
    })


def test_perfect_match(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Doe", "firstname": "Mary", "age": "9"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Brown", "firstname": "John", "age": "8"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "Doe", "firstname": "Mary", "age": "9"},
        {"lastname": "Brown", "firstname": "John", "age": "8"},
    ]
    pred = make_pred("doc1", header, pred_rows)

    report = evaluate_predictions([pred], gt_path)
    doc = report["documents"][0]
    assert doc["row_matches"] == [(0, 0), (1, 1)]
    assert doc["row_pred_unmatched"] == []
    assert doc["row_exp_unmatched"] == []


def test_extra_predicted_row(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Doe", "firstname": "Mary", "age": "9"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "Doe", "firstname": "Mary", "age": "9"},
        {"lastname": "X", "firstname": "Y", "age": "7"},
    ]
    pred = make_pred("doc1", header, pred_rows)

    report = evaluate_predictions([pred], gt_path)
    doc = report["documents"][0]
    assert doc["row_matches"] == [(0, 0)]
    assert doc["row_pred_unmatched"] == [1]
    assert doc["row_exp_unmatched"] == []


def test_missing_predicted_row(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Doe", "firstname": "Mary", "age": "9"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Brown", "firstname": "John", "age": "8"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "Doe", "firstname": "Mary", "age": "9"},
    ]
    pred = make_pred("doc1", header, pred_rows)

    report = evaluate_predictions([pred], gt_path)
    doc = report["documents"][0]
    assert doc["row_matches"] == [(0, 0)]
    assert doc["row_pred_unmatched"] == []
    assert len(doc["row_exp_unmatched"]) == 1


def test_duplicate_names(tmp_path: Path):
    # create ground truth with two identical rows
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Lee", "firstname": "Sam", "age": "10"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Lee", "firstname": "Sam", "age": "10"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    pred_rows = [
        {"lastname": "Lee", "firstname": "Sam", "age": "10"},
        {"lastname": "Lee", "firstname": "Sam", "age": "10"},
    ]
    pred = make_pred("doc1", header, pred_rows)

    report = evaluate_predictions([pred], gt_path)
    doc = report["documents"][0]
    assert doc["row_matches"] == [(0, 0), (1, 1)]
    assert doc["row_pred_unmatched"] == []
    assert doc["row_exp_unmatched"] == []
