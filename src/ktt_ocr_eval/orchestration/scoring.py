from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ktt_ocr_eval.core.artifacts import read_predictions
from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.evaluation.report import evaluate_predictions

VariantKey = tuple[str, str]  # (method name, config fingerprint)

# Which summary values become their own metric rows.
_SCALAR_METRICS = ("precision", "recall", "f1")
_COUNT_METRICS = ("tp", "fp", "fn", "document_count", "predicted_rows", "expected_rows")
_FAILURE_METRICS = {
    "collected": "records_collected",
    "succeeded": "records_succeeded",
    "failed": "records_failed",
    "failure_rate": "failure_rate",
}


def load_predictions_tree(path: Path) -> list[OCRPrediction]:
    """Load predictions from a single jsonl file, or every *.jsonl in a folder."""
    if path.is_dir():
        predictions: list[OCRPrediction] = []
        for file in sorted(path.glob("*.jsonl")):
            predictions.extend(read_predictions(file))
        return predictions
    return read_predictions(path)


def is_failure(prediction: OCRPrediction) -> bool:
    """True for a failed-call record (collector stored metadata.error). Excluded from
    the metric numerator/denominator and reported as a separate failure rate."""
    return bool(prediction.metadata.get("error"))


def variant_key(prediction: OCRPrediction) -> VariantKey:
    fingerprint = prediction.collection.fingerprint if prediction.collection else ""
    return (prediction.method, fingerprint)


def variant_label(prediction: OCRPrediction) -> str:
    nickname = prediction.collection.nickname if prediction.collection else None
    if nickname:
        return nickname
    method, fingerprint = variant_key(prediction)
    return f"{method}@{fingerprint[:8]}" if fingerprint else method


def group_by_variant(
    predictions: Iterable[OCRPrediction],
) -> dict[VariantKey, list[OCRPrediction]]:
    groups: dict[VariantKey, list[OCRPrediction]] = {}
    for prediction in predictions:
        groups.setdefault(variant_key(prediction), []).append(prediction)
    return groups


def score_suite(
    predictions_path: Path,
    ground_truth_path: Path,
    out_dir: Path,
    config: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """The middle "scores" stage: emit a tidy ``scores-<runid>.jsonl``.

    Group predictions into (method, config) variants, drop failed-call records
    (kept as a separate failure-rate metric), score each variant via
    ``report.evaluate_predictions`` (unchanged), and write one row per
    (variant x metric) — each row self-describing with its method, fingerprint,
    nickname, and config, mirroring the predictions jsonl. Ranking / presentation
    is the separate ``render`` stage.
    """
    predictions = load_predictions_tree(predictions_path)
    groups = group_by_variant(predictions)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    scores_path = out_dir / f"scores-{run_id}.jsonl"

    rows: list[dict[str, Any]] = []
    with scores_path.open("w", encoding="utf-8") as out:
        for variant in groups.values():
            clean = [p for p in variant if not is_failure(p)]
            failed = [p for p in variant if is_failure(p)]
            report = evaluate_predictions(clean, ground_truth_path, config)
            summary = report["summary"]
            summary["failures"] = {
                "collected": len(variant),
                "succeeded": len(clean),
                "failed": len(failed),
                "failure_rate": round(len(failed) / len(variant), 4) if variant else 0.0,
            }
            variant_rows = _summary_to_rows(variant[0], summary)
            for row in variant_rows:
                out.write(json.dumps(row))
                out.write("\n")
            out.flush()  # incremental: a crash keeps the variants already scored
            rows.extend(variant_rows)
    return scores_path, rows


def _identity(prediction: OCRPrediction) -> dict[str, Any]:
    collection = prediction.collection
    return {
        "method": prediction.method,
        "fingerprint": collection.fingerprint if collection else "",
        "nickname": collection.nickname if collection else None,
        "config": collection.config if collection else {},
    }


def _summary_to_rows(prediction: OCRPrediction, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one variant's report summary into tidy one-metric-per-row records."""
    identity = _identity(prediction)
    rows: list[dict[str, Any]] = []

    def emit(metric: str, value: Any, field: str | None = None) -> None:
        rows.append({**identity, "metric": metric, "field": field, "value": value})

    for metric in _SCALAR_METRICS:
        emit(metric, summary.get(metric, 0.0))
    for metric in _COUNT_METRICS:
        if metric in summary:
            emit(metric, summary[metric])
    for field, value in summary.get("field_accuracy", {}).items():
        emit("field_accuracy", value, field=field)
    failures = summary.get("failures", {})
    for key, name in _FAILURE_METRICS.items():
        if key in failures:
            emit(name, failures[key])
    return rows
