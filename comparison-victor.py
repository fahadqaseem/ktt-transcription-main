from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import TypedDict

from ktt_ocr_eval.core.columns import DEFAULT_ROW_MATCH_COLUMNS
from ktt_ocr_eval.evaluation.normalize import normalize_value, tokens


class MatchResult(TypedDict):
    matches: list[tuple[int, int]]
    pred_unmatched: list[int]
    exp_unmatched: list[int]


def composite_key(record: dict[str, str], key_fields: list[str] | None = None) -> str:
    """Create a deterministic composite key from a row using selected fields.

    Normalizes values before joining so small formatting differences don't
    affect exact-key matching.
    """
    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS
    parts: list[str] = []
    for field in key_fields:
        parts.append(normalize_value(record.get(field, "")))
    return "||".join(parts)


def match_by_exact_key(
    predicted_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    key_fields: list[str] | None = None,
) -> MatchResult:
    """Match predicted rows to expected rows using exact composite keys.

    Returns a dict containing:
      - matches: list of (pred_idx, exp_idx)
      - pred_unmatched: list of predicted indices that had no match
      - exp_unmatched: list of expected indices that were not matched

    Matching strategy: for expected rows build a mapping key -> queue of
    expected indices. For each predicted row, compute key and pop the first
    available expected index for that key (FIFO). This handles duplicates
    deterministically.
    """
    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS

    exp_map: dict[str, list[int]] = {}
    for i, row in enumerate(expected_rows):
        key = composite_key(row, key_fields)
        exp_map.setdefault(key, []).append(i)

    matches: list[tuple[int, int]] = []
    pred_unmatched: list[int] = []

    for pi, prow in enumerate(predicted_rows):
        key = composite_key(prow, key_fields)
        bucket = exp_map.get(key)
        if bucket:
            ei = bucket.pop(0)
            matches.append((pi, ei))
            if not bucket:
                del exp_map[key]
        else:
            pred_unmatched.append(pi)

    # Remaining expected indices are unmatched
    exp_unmatched: list[int] = []
    for indices in exp_map.values():
        exp_unmatched.extend(indices)

    return {
        "matches": matches,
        "pred_unmatched": pred_unmatched,
        "exp_unmatched": exp_unmatched,
    }


def jaccard_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union > 0 else 0.0


def row_token_set(record: dict[str, str], key_fields: list[str]) -> set[str]:
    parts: list[str] = []
    for field in key_fields:
        parts.extend(tokens(record.get(field, "")))
    return set(parts)


def match_by_jaccard(
    predicted_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    key_fields: list[str] | None = None,
    threshold: float = 0.5,
) -> MatchResult:
    """Match predicted to expected rows using token Jaccard similarity and Scipy.

    Uses scipy.optimize.linear_sum_assignment to find the optimal one-to-one
    matching that minimizes total cost (1.0 - Jaccard score). Matches falling
    below `threshold` are rejected.
    """
    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS

    predicted_token_sets = [row_token_set(row, key_fields) for row in predicted_rows]
    expected_token_sets = [row_token_set(row, key_fields) for row in expected_rows]
    
    # Build the Cost Matrix
    # Scipy minimizes cost, so a perfect Jaccard score of 1.0 becomes a cost of 0.0
    cost_matrix = []
    for predicted_tokens in predicted_token_sets:
        row_costs = []
        for expected_tokens in expected_token_sets:
            score = jaccard_similarity(predicted_tokens, expected_tokens)
            # Apply threshold: if score is too low, make the cost infinitely high
            if score >= threshold:
                row_costs.append(1.0 - score) 
            else:
                row_costs.append(float('inf'))
        cost_matrix.append(row_costs)
        
    # Run the Hungarian Algorithm using Scipy
    cost_array = np.array(cost_matrix)
    matches: list[tuple[int, int]] = []
    
    if cost_array.size > 0:
        row_ind, col_ind = linear_sum_assignment(cost_array)
        
        # Filter out matches that hit our 'inf' threshold penalty
        for r, c in zip(row_ind, col_ind):
            if cost_array[r, c] != float('inf'):
                matches.append((int(r), int(c)))

    # Categorize the unmatched indices
    matched_pred = {pred_index for pred_index, _ in matches}
    matched_exp = {exp_index for _, exp_index in matches}
    
    pred_unmatched = [
        pred_index for pred_index in range(len(predicted_rows)) if pred_index not in matched_pred
    ]
    exp_unmatched = [
        exp_index for exp_index in range(len(expected_rows)) if exp_index not in matched_exp
    ]
    
    return {"matches": matches, "pred_unmatched": pred_unmatched, "exp_unmatched": exp_unmatched}


def match_rows(
    predicted_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    key_fields: list[str] | None = None,
    strategy: str = "exact",
    threshold: float = 0.5,
) -> MatchResult:
    if strategy == "exact":
        return match_by_exact_key(predicted_rows, expected_rows, key_fields=key_fields)
    if strategy == "jaccard":
        return match_by_jaccard(
            predicted_rows,
            expected_rows,
            key_fields=key_fields,
            threshold=threshold,
        )
    raise ValueError(f"Unknown match strategy: {strategy}")
