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


def test_jaccard_matches_typo(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Smith", "firstname": "Sam", "age": "10"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    # predicted has extra token (junior) but shares token Smith
    pred_rows = [{"lastname": "Smith Jr", "firstname": "Sam", "age": "10"}]

    report = evaluate_predictions(
        [make_pred("d", header, pred_rows)],
        gt_path,
        config={"match_key_fields": ["lastname"], "match_strategy": "jaccard", "match_threshold": 0.3},
    )
    doc = report["documents"][0]
    # should match despite punctuation difference
    assert doc["row_matches"] == [(0, 0)]
