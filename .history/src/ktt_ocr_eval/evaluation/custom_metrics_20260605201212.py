from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ktt_ocr_eval.evaluation.example_metric import example_completeness_metric

CustomMetric = Callable[[dict[str, Any], dict[str, Any]], dict[str, float | int | str]]

CUSTOM_METRICS: dict[str, CustomMetric] = {
    "example_completeness": example_completeness_metric,
}


def evaluate_custom_metrics(
    context: dict[str, Any],
    metric_configs: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | str]]:
    results: dict[str, dict[str, float | int | str]] = {}
    for metric_config in metric_configs:
        metric_key = metric_config["metric"]
        metric_name = metric_config.get("name", metric_key)
        metric = get_custom_metric(metric_key)
        results[metric_name] = metric(context, metric_config.get("config", {}))
    return results


def get_custom_metric(metric_key: str) -> CustomMetric:
    try:
        return CUSTOM_METRICS[metric_key]
    except KeyError as exc:
        supported = ", ".join(sorted(CUSTOM_METRICS))
        message = f"Unknown custom metric {metric_key!r}. Supported metrics: {supported}"
        raise ValueError(message) from exc
