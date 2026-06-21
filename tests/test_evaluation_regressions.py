from __future__ import annotations

import csv
from pathlib import Path

from ktt_ocr_eval.core.columns import ALL_COLUMNS
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.report import evaluate_predictions


def make_pred(
    doc_id: str,
    header: dict[str, str],
    rows: list[dict[str, str]],
) -> OCRPrediction:
    return OCRPrediction.model_validate(
        {
            "document_id": doc_id,
            "method": "m",
            "header": header,
            "rows": rows,
        }
    )


def write_gt(path: Path, rows: list[dict[str, str]]) -> None:
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(ALL_COLUMNS))
    keys = [*ALL_COLUMNS, *extra_keys]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def test_unmatched_rows_reduce_field_accuracy_and_cer(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.csv"
    write_gt(
        gt_path,
        [
            {
                "school": "S",
                "grade": "1",
                "teacher": "T",
                "lastname": "Doe",
                "firstname": "Mary",
                "age": "9",
            }
        ],
    )

    report = evaluate_predictions(
        [
            make_pred(
                "doc1",
                {"school": "S", "grade": "1", "teacher": "T"},
                [{"lastname": "DoeX", "firstname": "Mary", "age": "9"}],
            )
        ],
        gt_path,
    )

    document = report["documents"][0]
    assert document["row_matches"] == []
    assert document["field_metrics"]["lastname"] == {
        "correct": 0,
        "total": 2,
        "accuracy": 0.0,
    }
    assert document["cer_metrics"]["lastname"] == {
        "char_errors": 7,
        "total_chars": 7,
        "cer": 1.0,
    }


def test_ground_truth_uses_document_id_when_available(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.csv"
    write_gt(
        gt_path,
        [
            {
                "document_id": "doc1",
                "school": "S",
                "grade": "1",
                "teacher": "T",
                "lastname": "Doe",
                "firstname": "Mary",
                "age": "9",
            },
            {
                "document_id": "doc2",
                "school": "S",
                "grade": "1",
                "teacher": "T",
                "lastname": "Brown",
                "firstname": "John",
                "age": "8",
            },
        ],
    )

    report = evaluate_predictions(
        [
            make_pred(
                "doc2",
                {"school": "S", "grade": "1", "teacher": "T"},
                [{"lastname": "Brown", "firstname": "John", "age": "8"}],
            )
        ],
        gt_path,
    )

    document = report["documents"][0]
    assert document["expected_row_count"] == 1
    assert document["row_matches"] == [(0, 0)]
    assert document["row_exp_unmatched"] == []


def test_jaccard_matching_uses_best_global_assignment(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.csv"
    write_gt(
        gt_path,
        [
            {
                "school": "S",
                "grade": "1",
                "teacher": "T",
                "lastname": "alpha beta",
                "firstname": "X",
                "age": "1",
            },
            {
                "school": "S",
                "grade": "1",
                "teacher": "T",
                "lastname": "alpha gamma",
                "firstname": "Y",
                "age": "2",
            },
        ],
    )

    report = evaluate_predictions(
        [
            make_pred(
                "doc1",
                {"school": "S", "grade": "1", "teacher": "T"},
                [
                    {"lastname": "alpha beta gamma", "firstname": "X", "age": "1"},
                    {"lastname": "alpha beta", "firstname": "Y", "age": "2"},
                ],
            )
        ],
        gt_path,
        config={
            "match_key_fields": ["lastname"],
            "match_strategy": "jaccard",
            "match_threshold": 0.5,
        },
    )

    document = report["documents"][0]
    assert document["row_matches"] == [(0, 1), (1, 0)]
    assert document["row_pred_unmatched"] == []
    assert document["row_exp_unmatched"] == []
