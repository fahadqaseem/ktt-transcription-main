from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ktt_ocr_eval.core.columns import ALL_COLUMNS, HEADER_COLUMNS
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.custom_metrics import evaluate_custom_metrics
from ktt_ocr_eval.evaluation.comparison import match_by_exact_key
from ktt_ocr_eval.core.columns import ROW_COLUMNS
from ktt_ocr_eval.evaluation.normalize import normalize_value
from ktt_ocr_eval.evaluation.normalize import normalize_value


def evaluate_predictions(
    predictions: list[OCRPrediction],
    ground_truth_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    ground_truth = read_ground_truth(ground_truth_path)
    filter_fields = config.get("header_filter_fields", HEADER_COLUMNS)
    custom_metric_configs = config.get("custom_metrics", [])
    evaluated_fields = config.get("evaluated_fields", ROW_COLUMNS)
    cer_fields = config.get("cer_fields", evaluated_fields)

    documents = [
        evaluate_prediction(
            prediction,
            ground_truth,
            filter_fields=filter_fields,
            custom_metric_configs=custom_metric_configs,
            evaluated_fields=evaluated_fields,
            cer_fields=cer_fields,
        )
        for prediction in predictions
    ]
    return {"summary": summarize_documents(documents), "documents": documents}


def read_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(ALL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return frame


def evaluate_prediction(
    prediction: OCRPrediction,
    ground_truth: pd.DataFrame,
    *,
    filter_fields: list[str],
    custom_metric_configs: list[dict[str, Any]],
    evaluated_fields: list[str],
    cer_fields: list[str],
) -> dict[str, Any]:
    expected_rows, filter_status = filter_ground_truth(ground_truth, prediction, filter_fields)
    predicted_rows = prediction.expanded_records()
    custom_metrics = evaluate_custom_metrics(
        {
            "prediction": prediction,
            "predicted_rows": predicted_rows,
            "expected_rows": expected_rows,
            "filter_status": filter_status,
        },
        custom_metric_configs,
    )
    # row matching (phase 1): exact composite-key matching
    row_matching = match_by_exact_key(predicted_rows, expected_rows)
    # per-document TP/FP/FN and PRF
    tp = len(row_matching.get("matches", []))
    fp = len(row_matching.get("pred_unmatched", []))
    fn = len(row_matching.get("exp_unmatched", []))
    doc_precision, doc_recall, doc_f1 = compute_prf(tp, fp, fn)
    # field-level scoring (phase 3): compute per-field correctness on matched rows
    field_metrics = compute_field_metrics(
        row_matching.get("matches", []), predicted_rows, expected_rows, evaluated_fields
    )
    # character-error-rate for selected fields
    cer_metrics = compute_cer_metrics(
        row_matching.get("matches", []), predicted_rows, expected_rows, cer_fields
    )
    return {
        "document_id": prediction.document_id,
        "method": prediction.method,
        "filter_status": filter_status,
        "header": prediction.header.canonical(),
        "predicted_row_count": len(predicted_rows),
        "expected_row_count": len(expected_rows),
        "custom_metrics": custom_metrics,
        "row_matches": row_matching["matches"],
        "row_pred_unmatched": row_matching["pred_unmatched"],
        "row_exp_unmatched": row_matching["exp_unmatched"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": doc_precision,
        "recall": doc_recall,
        "f1": doc_f1,
        "field_metrics": field_metrics,
        "cer_metrics": cer_metrics,
        "metadata": prediction.metadata,
    }


def filter_ground_truth(
    ground_truth: pd.DataFrame,
    prediction: OCRPrediction,
    filter_fields: list[str],
) -> tuple[list[dict[str, str]], str]:
    header = prediction.header.canonical()
    active_fields = [field for field in filter_fields if normalize_value(header.get(field, ""))]
    if not active_fields:
        return [], "no_header_values"

    mask = pd.Series([True] * len(ground_truth), index=ground_truth.index)
    for field in active_fields:
        predicted_value = normalize_value(header[field])
        mask &= ground_truth[field].map(normalize_value) == predicted_value

    rows = ground_truth.loc[mask, ALL_COLUMNS].to_dict(orient="records")
    status = "matched" if rows else "no_ground_truth_match"
    return [{key: str(value) for key, value in row.items()} for row in rows], status


def summarize_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    total_predicted = sum(int(document["predicted_row_count"]) for document in documents)
    total_expected = sum(int(document["expected_row_count"]) for document in documents)
    total_tp = sum(len(document.get("row_matches", [])) for document in documents)
    total_fp = sum(len(document.get("row_pred_unmatched", [])) for document in documents)
    total_fn = sum(len(document.get("row_exp_unmatched", [])) for document in documents)

    precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)

    # aggregate field-level totals across documents
    field_totals: dict[str, dict[str, int]] = {}
    for document in documents:
        fm = document.get("field_metrics", {})
        for field, stats in fm.items():
            if field not in field_totals:
                field_totals[field] = {"correct": 0, "total": 0}
            field_totals[field]["correct"] += int(stats.get("correct", 0))
            field_totals[field]["total"] += int(stats.get("total", 0))

    field_accuracy: dict[str, float] = {}
    for field, stats in field_totals.items():
        if stats["total"] == 0:
            field_accuracy[field] = 0.0
        else:
            field_accuracy[field] = round(stats["correct"] / stats["total"], 4)

    return {
        "document_count": len(documents),
        "predicted_rows": total_predicted,
        "expected_rows": total_expected,
        "filter_status_counts": count_values(document["filter_status"] for document in documents),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_accuracy": field_accuracy,
    }


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 with safe division and rounding.

    Returns (precision, recall, f1) rounded to 4 decimals.
    """
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_field_metrics(
    matches: list[tuple[int, int]],
    predicted_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    fields: list[str],
) -> dict[str, dict[str, int | float]]:
    """Compute per-field correct/total/accuracy over matched row pairs.

    - `matches` is list of (pred_idx, exp_idx) pairs returned by matcher.
    - `fields` is the list of row fields to evaluate.
    Returns a mapping field -> {"correct": int, "total": int, "accuracy": float}
    """
    results: dict[str, dict[str, int | float]] = {}
    # initialize counts
    for field in fields:
        results[field] = {"correct": 0, "total": 0, "accuracy": 0.0}

    for pi, ei in matches:
        pred = predicted_rows[pi]
        exp = expected_rows[ei]
        for field in fields:
            results[field]["total"] += 1
            pval = normalize_value(pred.get(field, ""))
            eval = normalize_value(exp.get(field, ""))
            if pval == eval:
                results[field]["correct"] += 1

    for field in fields:
        total = int(results[field]["total"])
        correct = int(results[field]["correct"])
        acc = round(correct / total, 4) if total > 0 else 0.0
        results[field]["accuracy"] = acc

    return results


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report["summary"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "documents.json").write_text(
        json.dumps(report["documents"], indent=2),
        encoding="utf-8",
    )
    write_documents_csv(report["documents"], out_dir / "documents.csv")


def write_documents_csv(documents: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = {
            "method": document["method"],
            "document_id": document["document_id"],
            "filter_status": document["filter_status"],
            "predicted_row_count": document["predicted_row_count"],
            "expected_row_count": document["expected_row_count"],
        }
        for metric_name, metrics in document.get("custom_metrics", {}).items():
            for metric, value in metrics.items():
                row[f"custom_{metric_name}_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
