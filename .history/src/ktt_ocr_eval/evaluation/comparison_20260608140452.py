from __future__ import annotations

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


def assignment_maximizing_similarity(scores: list[list[float]]) -> list[tuple[int, int]]:
    """Return the highest-scoring one-to-one assignment using Hungarian algorithm."""
    if not scores or not scores[0]:
        return []

    row_count = len(scores)
    col_count = len(scores[0])
    size = max(row_count, col_count)
    padded_scores = [[0.0] * size for _ in range(size)]
    for row_index, row in enumerate(scores):
        for col_index, score in enumerate(row):
            padded_scores[row_index][col_index] = score

    max_score = max(max(row) for row in padded_scores)
    costs = [[max_score - score for score in row] for row in padded_scores]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for row_index in range(1, size + 1):
        p[0] = row_index
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        column_index = 0
        while True:
            used[column_index] = True
            current_row = p[column_index]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = (
                    costs[current_row - 1][candidate_column - 1]
                    - u[current_row]
                    - v[candidate_column]
                )
                if current < minv[candidate_column]:
                    minv[candidate_column] = current
                    way[candidate_column] = column_index
                if minv[candidate_column] < delta:
                    delta = minv[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minv[candidate_column] -= delta
            column_index = next_column
            if p[column_index] == 0:
                break
        while True:
            next_column = way[column_index]
            p[column_index] = p[next_column]
            column_index = next_column
            if column_index == 0:
                break

    matches: list[tuple[int, int]] = []
    for col_index in range(1, size + 1):
        row_index = p[col_index]
        if row_index == 0:
            continue
        pred_index = row_index - 1
        exp_index = col_index - 1
        if pred_index < row_count and exp_index < col_count:
            matches.append((pred_index, exp_index))
    matches.sort()
    return matches


def match_by_jaccard(
    predicted_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    key_fields: list[str] | None = None,
    threshold: float = 0.5,
) -> MatchResult:
    """Match predicted to expected rows using token Jaccard similarity on composite key fields.

    For each predicted row, find the highest-Jaccard expected row above `threshold`.
    Matches are one-to-one and expected rows are consumed when matched.
    """
    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS

    predicted_token_sets = [row_token_set(row, key_fields) for row in predicted_rows]
    expected_token_sets = [row_token_set(row, key_fields) for row in expected_rows]
    scores = [
        [
            score
            if (score := jaccard_similarity(predicted_tokens, expected_tokens)) >= threshold
            else 0.0
            for expected_tokens in expected_token_sets
        ]
        for predicted_tokens in predicted_token_sets
    ]

    assigned_pairs = assignment_maximizing_similarity(scores)
    matches = [
        (pred_index, exp_index)
        for pred_index, exp_index in assigned_pairs
        if scores[pred_index][exp_index] >= threshold
    ]
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
