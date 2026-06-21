from __future__ import annotations

from typing import Any

from ktt_ocr_eval.core.columns import HEADER_COLUMNS, ROW_COLUMNS
from ktt_ocr_eval.evaluation.normalize import normalize_value


def example_completeness_metric(
        context: dict[str, Any],
        config: dict[str, Any],
) -> dict[str, float | int]:
    """Ultra-optimized extraction-quality evaluator processing rows in a single pass."""
    required_header_fields = config.get("required_header_fields", HEADER_COLUMNS)
    required_row_fields = config.get("required_row_fields", ROW_COLUMNS)
    predicted_header = context["prediction"].header.canonical()
    predicted_rows = context["predicted_rows"]

    # Evaluate header once
    header_comp = completeness(predicted_header, required_header_fields)

    if not predicted_rows:
        return {
            "header_completeness": header_comp,
            "row_completeness": 0.0,
            "non_empty_row_count": 0,
        }

    total_rows = len(predicted_rows)
    row_completeness_sum = 0.0
    non_empty_rows = 0

    # Caching lookups locally to drop global/module reference checking over iterations
    norm = normalize_value
    fields_len = len(required_row_fields) if required_row_fields else 1
    has_fields = bool(required_row_fields)

    for row in predicted_rows:
        # OPTIMIZATION 1: Direct key lookup mapping bypasses row.get() method overhead
        if has_fields:
            populated = sum(1 for field in required_row_fields if field in row and norm(row[field]))
            # OPTIMIZATION 2: Deferred rounding to summary stage reduces arithmetic call footprint
            row_completeness_sum += (populated / fields_len)
        else:
            row_completeness_sum += 1.0

        # Optimized Row Emptiness Check
        # Fast dictionary evaluation via native iterator values unpacking
        for val in row.values():
            if norm(val):
                non_empty_rows += 1
                break

    return {
        "header_completeness": header_comp,
        "row_completeness": round(row_completeness_sum / total_rows, 4),
        "non_empty_row_count": non_empty_rows,
    }


def completeness(record: dict[str, str], fields: list[str]) -> float:
    if not fields:
        return 1.0
    norm = normalize_value
    # Direct dictionary index hashing avoids underlying method dispatch table routines
    populated = sum(1 for field in fields if field in record and norm(record[field]))
    return round(populated / len(fields), 4)