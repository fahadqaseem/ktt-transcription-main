from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ktt_ocr_eval.evaluation.example_metric import example_completeness_metric

MetricValue = float | int | str
MetricResult = Mapping[str, MetricValue]
CustomMetric = Callable[[dict[str, Any], dict[str, Any]], MetricResult]

# Explicit registry mapping
CUSTOM_METRICS: dict[str, CustomMetric] = {
    "example_completeness": example_completeness_metric,
}


def evaluate_custom_metrics(
        context: dict[str, Any],
        metric_configs: list[dict[str, Any]],
) -> dict[str, dict[str, MetricValue]]:
    """Streamlined custom evaluator routing engine."""
    results: dict[str, dict[str, MetricValue]] = {}
    metrics_map = CUSTOM_METRICS  # Local pointer variable optimization

    for metric_config in metric_configs:
        # Extract variables using fast key access
        metric_key = metric_config["metric"]
        metric_name = metric_config.get("name", metric_key)

        if metric_key not in metrics_map:
            supported = ", ".join(sorted(metrics_map.keys()))
            raise ValueError(f"Unknown custom metric {metric_key!r}. Supported metrics: {supported}")

        # Execute metric directly. Removes dict() wrapper casting to preserve references natively.
        results[metric_name] = metrics_map[metric_key](context, metric_config.get("config", {}))

    return results