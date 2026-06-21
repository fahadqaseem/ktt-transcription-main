from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from ktt_ocr_eval.core.columns import ALL_COLUMNS, HEADER_COLUMNS, ROW_COLUMNS
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.comparison import MatchResult, match_rows
from ktt_ocr_eval.evaluation.custom_metrics import MetricValue, evaluate_custom_metrics
from ktt_ocr_eval.evaluation.normalize import normalize_value

RowPair = tuple[int | None, int | None]
Record = dict[str, str]
MetricStats = dict[str, int | float]
CerStats = dict[str, int | float]
CustomMetricResults = dict[str, dict[str, MetricValue]]


class DocumentReport(TypedDict):
    document_id: str
    method: str
    filter_status: str
    header: Record
    predicted_row_count: int
    expected_row_count: int
    custom_metrics: CustomMetricResults
    row_matches: list[tuple[int, int]]
    row_pred_unmatched: list[int]
    row_exp_unmatched: list[int]
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    field_metrics: dict[str, MetricStats]
    cer_metrics: dict[str, CerStats]
    metadata: dict[str, Any]


class MethodSummary(TypedDict):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    field_accuracy: dict[str, float]


SummaryReport = dict[str, Any]


class EvaluationReport(TypedDict):
    summary: SummaryReport
    documents: list[DocumentReport]


def evaluate_predictions(
    predictions: list[OCRPrediction],
    ground_truth_path: Path,
    config: dict[str, Any] | None = None,
) -> EvaluationReport:
    config = config or {}
    ground_truth = read_ground_truth(ground_truth_path)
    filter_fields = _coerce_string_list(config.get("header_filter_fields"), HEADER_COLUMNS)
    custom_metric_configs = _coerce_metric_configs(config.get("custom_metrics"))
    evaluated_fields = _coerce_string_list(config.get("evaluated_fields"), ROW_COLUMNS)
    cer_fields = _coerce_string_list(config.get("cer_fields"), evaluated_fields)
    match_key_fields = _coerce_optional_string_list(config.get("match_key_fields"))
    match_strategy = _coerce_match_strategy(config.get("match_strategy"))
    match_threshold = _coerce_match_threshold(config.get("match_threshold"))

    documents: list[DocumentReport] = [
        evaluate_prediction(
            prediction,
            ground_truth,
            filter_fields=filter_fields,
            custom_metric_configs=custom_metric_configs,
            evaluated_fields=evaluated_fields,
            cer_fields=cer_fields,
            match_key_fields=match_key_fields,
            match_strategy=match_strategy,
            match_threshold=match_threshold,
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
    match_key_fields: list[str] | None = None,
    match_strategy: str = "exact",
    match_threshold: float = 0.5,
) -> DocumentReport:
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
    row_matching: MatchResult = match_rows(
        predicted_rows,
        expected_rows,
        key_fields=match_key_fields,
        strategy=match_strategy,
        threshold=match_threshold,
    )
    # per-document TP/FP/FN and PRF
    matches = row_matching["matches"]
    pred_unmatched = row_matching["pred_unmatched"]
    exp_unmatched = row_matching["exp_unmatched"]
    tp = len(matches)
    fp = len(pred_unmatched)
    fn = len(exp_unmatched)
    doc_precision, doc_recall, doc_f1 = compute_prf(tp, fp, fn)
    # field-level scoring (phase 3): score matched and unmatched row alignments
    field_metrics = compute_field_metrics(
        matches,
        pred_unmatched,
        exp_unmatched,
        predicted_rows,
        expected_rows,
        evaluated_fields,
    )
    # character-error-rate for selected fields
    cer_metrics = compute_cer_metrics(
        matches,
        pred_unmatched,
        exp_unmatched,
        predicted_rows,
        expected_rows,
        cer_fields,
    )
    return {
        "document_id": prediction.document_id,
        "method": prediction.method,
        "filter_status": filter_status,
        "header": prediction.header.canonical(),
        "predicted_row_count": len(predicted_rows),
        "expected_row_count": len(expected_rows),
        "custom_metrics": custom_metrics,
        "row_matches": matches,
        "row_pred_unmatched": pred_unmatched,
        "row_exp_unmatched": exp_unmatched,
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
    document_mask = pd.Series([True] * len(ground_truth), index=ground_truth.index)
    if "document_id" in ground_truth.columns:
        document_mask &= ground_truth["document_id"].map(normalize_value) == normalize_value(
            prediction.document_id
        )
    if prediction.source_path and "source_path" in ground_truth.columns:
        document_mask &= ground_truth["source_path"].map(normalize_value) == normalize_value(
            prediction.source_path
        )

    header = prediction.header.canonical()
    active_fields = [field for field in filter_fields if normalize_value(header.get(field, ""))]
    if not active_fields:
        rows = ground_truth.loc[document_mask, ALL_COLUMNS].to_dict(orient="records")
        if rows:
            return [stringify_record(row) for row in rows], "matched"
        return [], "no_header_values"

    mask = document_mask.copy()
    for field in active_fields:
        predicted_value = normalize_value(header[field])
        mask &= ground_truth[field].map(normalize_value) == predicted_value

    rows = ground_truth.loc[mask, ALL_COLUMNS].to_dict(orient="records")
    status = "matched" if rows else "no_ground_truth_match"
    return [stringify_record(row) for row in rows], status


def summarize_documents(documents: list[DocumentReport]) -> SummaryReport:
    total_predicted = sum(document["predicted_row_count"] for document in documents)
    total_expected = sum(document["expected_row_count"] for document in documents)
    total_tp = sum(len(document["row_matches"]) for document in documents)
    total_fp = sum(len(document["row_pred_unmatched"]) for document in documents)
    total_fn = sum(len(document["row_exp_unmatched"]) for document in documents)

    precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)

    # aggregate field-level totals across documents
    field_totals: dict[str, dict[str, int]] = {}
    for document in documents:
        fm = document["field_metrics"]
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

    # aggregate per-method metrics
    methods: dict[str, MethodSummary] = {}
    for document in documents:
        method = document["method"]
        if method not in methods:
            methods[method] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "field_accuracy": {},
            }
        methods[method]["tp"] += document["tp"]
        methods[method]["fp"] += document["fp"]
        methods[method]["fn"] += document["fn"]

    # aggregate per-method field-level totals for accuracy
    method_field_totals: dict[str, dict[str, dict[str, int]]] = {}
    for document in documents:
        method = document["method"]
        fm = document["field_metrics"]
        if method not in method_field_totals:
            method_field_totals[method] = {}
        for field, stats in fm.items():
            if field not in method_field_totals[method]:
                method_field_totals[method][field] = {"correct": 0, "total": 0}
            method_field_totals[method][field]["correct"] += int(stats.get("correct", 0))
            method_field_totals[method][field]["total"] += int(stats.get("total", 0))

    # compute PRF per method
    for method, stats in methods.items():
        p, r, f = compute_prf(int(stats["tp"]), int(stats["fp"]), int(stats["fn"]))
        stats["precision"] = p
        stats["recall"] = r
        stats["f1"] = f
        # compute per-field accuracy for this method
        field_acc: dict[str, float] = {}
        for field, totals in method_field_totals.get(method, {}).items():
            if totals["total"] == 0:
                field_acc[field] = 0.0
            else:
                field_acc[field] = round(totals["correct"] / totals["total"], 4)
        stats["field_accuracy"] = field_acc

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
        "methods": methods,
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
    pred_unmatched: list[int],
    exp_unmatched: list[int],
    predicted_rows: list[Record],
    expected_rows: list[Record],
    fields: list[str],
) -> dict[str, MetricStats]:
    """Compute per-field correct/total/accuracy over aligned row pairs.

    Unmatched rows are counted as incorrect for every evaluated field so row-level
    misses and extras are reflected in field accuracy.
    - `fields` is the list of row fields to evaluate.
    Returns a mapping field -> {"correct": int, "total": int, "accuracy": float}
    """
    results: dict[str, MetricStats] = {}
    for field in fields:
        results[field] = {"correct": 0, "total": 0, "accuracy": 0.0}

    for pred_index, exp_index in build_row_pairs(matches, pred_unmatched, exp_unmatched):
        for field in fields:
            results[field]["total"] += 1
            if pred_index is None or exp_index is None:
                continue
            pval = normalize_value(predicted_rows[pred_index].get(field, ""))
            exp_val = normalize_value(expected_rows[exp_index].get(field, ""))
            if pval == exp_val:
                results[field]["correct"] += 1

    for field in fields:
        total = int(results[field]["total"])
        correct = int(results[field]["correct"])
        acc = round(correct / total, 4) if total > 0 else 0.0
        results[field]["accuracy"] = acc

    return results


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def compute_cer_metrics(
    matches: list[tuple[int, int]],
    pred_unmatched: list[int],
    exp_unmatched: list[int],
    predicted_rows: list[Record],
    expected_rows: list[Record],
    fields: list[str],
) -> dict[str, CerStats]:
    """Compute character error totals and CER over aligned row pairs.

    Returns mapping field -> {"char_errors": int, "total_chars": int, "cer": float}
    """
    results: dict[str, CerStats] = {}
    for field in fields:
        results[field] = {"char_errors": 0, "total_chars": 0, "cer": 0.0}

    for pred_index, exp_index in build_row_pairs(matches, pred_unmatched, exp_unmatched):
        for field in fields:
            if pred_index is None:
                pval = ""
            else:
                pval = normalize_value(predicted_rows[pred_index].get(field, ""))
            if exp_index is None:
                exp_val = ""
            else:
                exp_val = normalize_value(expected_rows[exp_index].get(field, ""))
            dist = levenshtein(pval, exp_val)
            total = max(len(exp_val), len(pval))
            results[field]["char_errors"] += dist
            results[field]["total_chars"] += total

    for field in fields:
        total = int(results[field]["total_chars"])
        errs = int(results[field]["char_errors"])
        cer = round(errs / total, 4) if total > 0 else 0.0
        results[field]["cer"] = cer

    return results


def build_row_pairs(
    matches: list[tuple[int, int]],
    pred_unmatched: list[int],
    exp_unmatched: list[int],
) -> list[RowPair]:
    pairs: list[RowPair] = list(matches)
    pairs.extend((pred_index, None) for pred_index in pred_unmatched)
    pairs.extend((None, exp_index) for exp_index in exp_unmatched)
    return pairs


def count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_report(report: EvaluationReport, out_dir: Path) -> None:
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


def write_documents_csv(documents: list[DocumentReport], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = {
            "method": document["method"],
            "document_id": document["document_id"],
            "filter_status": document["filter_status"],
            "predicted_row_count": document["predicted_row_count"],
            "expected_row_count": document["expected_row_count"],
        }
        for metric_name, metrics in document["custom_metrics"].items():
            for metric, value in metrics.items():
                row[f"custom_{metric_name}_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def stringify_record(row: Mapping[Hashable, Any]) -> Record:
    return {str(key): str(value) for key, value in row.items()}


def _coerce_string_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return list(default)


def _coerce_optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def _coerce_metric_configs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_match_strategy(value: Any) -> str:
    if value in {"exact", "jaccard"}:
        return value
    return "exact"


def _coerce_match_threshold(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.5
