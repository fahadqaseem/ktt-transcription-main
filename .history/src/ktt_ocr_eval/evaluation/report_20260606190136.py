from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ktt_ocr_eval.core.columns import ALL_COLUMNS, HEADER_COLUMNS
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.custom_metrics import evaluate_custom_metrics
from ktt_ocr_eval.evaluation.comparison import match_by_exact_key
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

    documents = [
        evaluate_prediction(
            prediction,
            ground_truth,
            filter_fields=filter_fields,
            custom_metric_configs=custom_metric_configs,
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
    return {
        "document_count": len(documents),
        "predicted_rows": sum(int(document["predicted_row_count"]) for document in documents),
        "expected_rows": sum(int(document["expected_row_count"]) for document in documents),
        "filter_status_counts": count_values(document["filter_status"] for document in documents),
    }


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
