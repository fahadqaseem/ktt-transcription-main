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


def test_per_document_prf_and_method_ranking(tmp_path: Path):
    gt = [
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "A", "firstname": "X", "age": "1"},
        {"school": "S", "grade": "1", "teacher": "T", "lastname": "B", "firstname": "Y", "age": "2"},
    ]
    gt_path = tmp_path / "gt.csv"
    write_gt(gt_path, gt)

    header = {"school": "S", "grade": "1", "teacher": "T"}
    # doc1: method m1 perfect
    p1 = make_pred("d1", "m1", header, [{"lastname": "A", "firstname": "X", "age": "1"}, {"lastname": "B", "firstname": "Y", "age": "2"}])
    # doc2: method m2 misses one
    p2 = make_pred("d2", "m2", header, [{"lastname": "A", "firstname": "X", "age": "1"}])

    report = evaluate_predictions([p1, p2], gt_path)
    docs = report["documents"]
    # per-document PRF present
    assert docs[0]["precision"] == 1.0
    assert docs[0]["recall"] == 1.0
    assert docs[1]["precision"] == 1.0
    assert docs[1]["recall"] == 0.5

    methods = report["summary"]["methods"]
    assert methods["m1"]["f1"] == 1.0
    # m2 has one TP, one FN -> recall 0.5
    assert methods["m2"]["recall"] == 0.5
