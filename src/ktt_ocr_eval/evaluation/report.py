from __future__ import annotations

import json
import concurrent.futures
from collections.abc import Hashable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

# Core and Local Module Imports
from ktt_ocr_eval.core.columns import ALL_COLUMNS, HEADER_COLUMNS, ROW_COLUMNS, ALL_COLUMNS_SET
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.comparison import MatchResult, match_rows
from ktt_ocr_eval.evaluation.custom_metrics import MetricValue, evaluate_custom_metrics
from ktt_ocr_eval.evaluation.normalize import normalize_value

# OPTIMIZATION 1: Fixed the module call error by cleanly separating function pointers
try:
    from Levenshtein import distance as levenshtein

    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


    # Pure Python fast fallback if native C-compiled binaries aren't bound
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if not s2:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

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


def _worker_evaluate(
        prediction: OCRPrediction,
        ground_truth_rows: list[dict[str, str]],
        *,
        filter_fields: list[str],
        custom_metric_configs: list[dict[str, Any]],
        evaluated_fields: list[str],
        cer_fields: list[str],
        match_key_fields: list[str] | None,
        match_strategy: str,
        match_threshold: float,
) -> DocumentReport:
    """Isolated worker function executing optimized metrics calculation over localized data chunks."""
    header = prediction.header.canonical()
    norm = normalize_value

    # Fast evaluation loop short-circuit for row filters
    active_fields = [f for f in filter_fields if f in header and norm(header[f])]

    expected_rows = ground_truth_rows
    filter_status = "matched"

    if not ground_truth_rows:
        filter_status = "no_ground_truth_match"
    elif active_fields:
        filtered = []
        for row in ground_truth_rows:
            if all(field in row and norm(row[field]) == norm(header[field]) for field in active_fields):
                filtered.append(row)
        expected_rows = filtered
        filter_status = "matched" if filtered else "no_ground_truth_match"

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

    matches = row_matching["matches"]
    pred_unmatched = row_matching["pred_unmatched"]
    exp_unmatched = row_matching["exp_unmatched"]

    tp, fp, fn = len(matches), len(pred_unmatched), len(exp_unmatched)
    doc_precision, doc_recall, doc_f1 = compute_prf(tp, fp, fn)

    row_pairs = list(matches)
    row_pairs.extend((p_idx, None) for p_idx in pred_unmatched)
    row_pairs.extend((None, e_idx) for e_idx in exp_unmatched)

    field_metrics = compute_field_metrics(
        row_pairs, predicted_rows, expected_rows, evaluated_fields
    )
    cer_metrics = compute_cer_metrics(
        row_pairs, predicted_rows, expected_rows, cer_fields
    )

    return {
        "document_id": prediction.document_id,
        "method": prediction.method,
        "filter_status": filter_status,
        "header": header,
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


def evaluate_predictions(
        predictions: list[OCRPrediction],
        ground_truth_path: Path,
        config: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Core entry point orchestrating parallel evaluation across local system cores."""
    config = config or {}
    ground_truth = read_ground_truth(ground_truth_path)

    filter_fields = _coerce_string_list(config.get("header_filter_fields"), HEADER_COLUMNS)
    custom_metric_configs = _coerce_metric_configs(config.get("custom_metrics"))
    evaluated_fields = _coerce_string_list(config.get("evaluated_fields"), ROW_COLUMNS)
    cer_fields = _coerce_string_list(config.get("cer_fields"), evaluated_fields)
    match_key_fields = _coerce_optional_string_list(config.get("match_key_fields"))
    match_strategy = _coerce_match_strategy(config.get("match_strategy"))
    match_threshold = _coerce_match_threshold(config.get("match_threshold"))

    norm = normalize_value
    has_doc_id = "document_id" in ground_truth.columns

    # ADAPTIVE ROUTING: Branch evaluation tracks based on document tracking availability
    if has_doc_id:
        gt_dict: dict[str, list[dict[str, str]]] = {}
        for doc_id, group in ground_truth.groupby("document_id", sort=False):
            raw_records = group.to_dict(orient="records")
            gt_dict[norm(str(doc_id))] = [
                {str(k): str(v) for k, v in r.items()} for r in raw_records
            ]
        tasks = [(pred, gt_dict.get(norm(pred.document_id), [])) for pred in predictions]
    else:
        raw_records = ground_truth.to_dict(orient="records")
        global_gt_rows = [
            {str(k): str(v) for k, v in r.items()} for r in raw_records
        ]
        tasks = [(pred, global_gt_rows) for pred in predictions]

    # Bind operational configs to worker closure payload
    worker_task = partial(
        _worker_evaluate,
        filter_fields=filter_fields,
        custom_metric_configs=custom_metric_configs,
        evaluated_fields=evaluated_fields,
        cer_fields=cer_fields,
        match_key_fields=match_key_fields,
        match_strategy=match_strategy,
        match_threshold=match_threshold,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_task, pred, gt_rows) for pred, gt_rows in tasks]
        # BUG FIX & OPTIMIZATION: Maintain original list indexing order instead of using as_completed
        documents = [f.result() for f in futures]

    return {"summary": summarize_documents(documents), "documents": documents}

def read_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = ALL_COLUMNS_SET - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return frame


def summarize_documents(documents: list[DocumentReport]) -> SummaryReport:
    total_predicted = sum(d["predicted_row_count"] for d in documents)
    total_expected = sum(d["expected_row_count"] for d in documents)
    total_tp = sum(len(d["row_matches"]) for d in documents)
    total_fp = sum(len(d["row_pred_unmatched"]) for d in documents)
    total_fn = sum(len(d["row_exp_unmatched"]) for d in documents)

    precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)

    field_totals: dict[str, dict[str, int]] = {}
    for document in documents:
        for field, stats in document["field_metrics"].items():
            totals = field_totals.setdefault(field, {"correct": 0, "total": 0})
            totals["correct"] += int(stats.get("correct", 0))
            totals["total"] += int(stats.get("total", 0))

    field_accuracy = {
        f: (round(st["correct"] / st["total"], 4) if st["total"] > 0 else 0.0)
        for f, st in field_totals.items()
    }

    methods: dict[str, MethodSummary] = {}
    method_field_totals: dict[str, dict[str, dict[str, int]]] = {}

    for document in documents:
        method = document["method"]
        if method not in methods:
            methods[method] = {
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "field_accuracy": {},
            }
        m_stat = methods[method]
        m_stat["tp"] += document["tp"]
        m_stat["fp"] += document["fp"]
        m_stat["fn"] += document["fn"]

        m_field = method_field_totals.setdefault(method, {})
        for field, stats in document["field_metrics"].items():
            totals = m_field.setdefault(field, {"correct": 0, "total": 0})
            totals["correct"] += int(stats.get("correct", 0))
            totals["total"] += int(stats.get("total", 0))

    for method, stats in methods.items():
        p, r, f = compute_prf(int(stats["tp"]), int(stats["fp"]), int(stats["fn"]))
        stats["precision"], stats["recall"], stats["f1"] = p, r, f
        stats["field_accuracy"] = {
            f: (round(t["correct"] / t["total"], 4) if t["total"] > 0 else 0.0)
            for f, t in method_field_totals.get(method, {}).items()
        }

    return {
        "document_count": len(documents),
        "predicted_rows": total_predicted,
        "expected_rows": total_expected,
        "filter_status_counts": count_values(d["filter_status"] for d in documents),
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_field_metrics(
        row_pairs: list[RowPair],
        predicted_rows: list[Record],
        expected_rows: list[Record],
        fields: list[str],
) -> dict[str, MetricStats]:
    """Compute per-field metrics utilizing precalculated alignment pairs."""
    results = {f: {"correct": 0, "total": 0, "accuracy": 0.0} for f in fields}
    norm = normalize_value

    for pred_index, exp_index in row_pairs:
        p_row = predicted_rows[pred_index] if pred_index is not None else None
        e_row = expected_rows[exp_index] if exp_index is not None else None

        for field in fields:
            res = results[field]
            res["total"] += 1
            if p_row is None or e_row is None:
                continue

            pval = p_row[field] if field in p_row else ""
            eval_val = e_row[field] if field in e_row else ""
            if norm(pval) == norm(eval_val):
                res["correct"] += 1

    for field in fields:
        res = results[field]
        tot = int(res["total"])
        res["accuracy"] = round(int(res["correct"]) / tot, 4) if tot > 0 else 0.0

    return results


def compute_cer_metrics(
        row_pairs: list[RowPair],
        predicted_rows: list[Record],
        expected_rows: list[Record],
        fields: list[str],
) -> dict[str, CerStats]:
    """Compute character error rate metrics utilizing precalculated alignment pairs."""
    results = {f: {"char_errors": 0, "total_chars": 0, "cer": 0.0} for f in fields}
    norm = normalize_value
    calc_dist = levenshtein

    for pred_index, exp_index in row_pairs:
        p_row = predicted_rows[pred_index] if pred_index is not None else None
        e_row = expected_rows[exp_index] if exp_index is not None else None

        for field in fields:
            pval = norm(p_row[field]) if (p_row and field in p_row) else ""
            exp_val = norm(e_row[field]) if (e_row and field in e_row) else ""

            res = results[field]
            res["char_errors"] += calc_dist(pval, exp_val)
            res["total_chars"] += max(len(exp_val), len(pval))

    for field in fields:
        res = results[field]
        tot = int(res["total_chars"])
        res["cer"] = round(int(res["char_errors"]) / tot, 4) if tot > 0 else 0.0

    return results


def count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    return counts


def write_report(report: EvaluationReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "summary.json").open("w", encoding="utf-8", buffering=64 * 1024) as f:
        f.write(json.dumps(report["summary"], indent=2))

    with (out_dir / "documents.json").open("w", encoding="utf-8", buffering=64 * 1024) as f:
        f.write(json.dumps(report["documents"], indent=2))

    write_documents_csv(report["documents"], out_dir / "documents.csv")


def write_documents_csv(documents: list[DocumentReport], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for doc in documents:
        row = {
            "method": doc["method"],
            "document_id": doc["document_id"],
            "filter_status": doc["filter_status"],
            "predicted_row_count": doc["predicted_row_count"],
            "expected_row_count": doc["expected_row_count"],
        }
        for metric_name, metrics in doc["custom_metrics"].items():
            for metric, value in metrics.items():
                row[f"custom_{metric_name}_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def stringify_record(row: Mapping[Hashable, Any]) -> Record:
    return {str(k): str(v) for k, v in row.items()}


def _coerce_string_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(i, str) for i in value):
        return list(value)
    return list(default)


def _coerce_optional_string_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(i, str) for i in value):
        return list(value)
    return None


def _coerce_metric_configs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [i for i in value if isinstance(i, dict)]


def _coerce_match_strategy(value: Any) -> str:
    return value if value in {"exact", "jaccard"} else "exact"


def _coerce_match_threshold(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.5

# This still needs changes... make it align with more data frameworks and types.... look into Levenshtein