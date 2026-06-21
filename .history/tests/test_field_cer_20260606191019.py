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


def test_cer_for_field(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "Ana", "firstname": "X", "age": "1"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    # small typo in lastname
    pred_rows = [{"lastname": "Ann", "firstname": "X", "age": "1"}]
    # request CER evaluation for lastname only
    report = evaluate_predictions([make_pred("d", header, pred_rows)], gt_path, config={"cer_fields": ["lastname"]})
    doc = report["documents"][0]
    cer = doc["cer_metrics"]["lastname"]
    # 'ana' vs 'ann' -> levenshtein distance 1, total chars 3 => cer = 0.3333
    assert cer["char_errors"] == 1
    assert cer["total_chars"] == 3
    assert abs(cer["cer"] - 0.3333) < 1e-4
