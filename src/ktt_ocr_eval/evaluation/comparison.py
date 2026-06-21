from __future__ import annotations

from typing import TypedDict
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

# Leverage our global O(1) frozen sets configuration
from ktt_ocr_eval.core.columns import DEFAULT_ROW_MATCH_COLUMNS
from ktt_ocr_eval.evaluation.normalize import normalize_value, tokens


class MatchResult(TypedDict):
    matches: list[tuple[int, int]]
    pred_unmatched: list[int]
    exp_unmatched: list[int]


def composite_key(record: dict[str, str], key_fields: list[str] | None = None) -> str:
    """Create a deterministic composite key from a row using selected fields."""
    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS

    norm = normalize_value
    # OPTIMIZATION 1: Direct hashing index reads avoid dispatch table routines
    parts: list[str] = [
        norm(record[field]) if field in record else ""
        for field in key_fields
    ]
    return "||".join(parts)


def match_by_exact_key(
        predicted_rows: list[dict[str, str]],
        expected_rows: list[dict[str, str]],
        key_fields: list[str] | None = None,
) -> MatchResult:
    """Match predicted rows to expected rows using exact composite keys."""
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

    exp_unmatched: list[int] = [
        idx for indices in exp_map.values() for idx in indices
    ]

    return {
        "matches": matches,
        "pred_unmatched": pred_unmatched,
        "exp_unmatched": exp_unmatched,
    }


def row_token_list(record: dict[str, str], key_fields: list[str]) -> list[str]:
    """Flattens unique tokens across fields into a list efficiently."""
    parts: list[str] = []
    tok = tokens

    # Direct hash tracking inside row parsing loop
    for field in key_fields:
        if field in record:
            parts.extend(tok(record[field]))

    return list(set(parts))


def match_by_jaccard(
        predicted_rows: list[dict[str, str]],
        expected_rows: list[dict[str, str]],
        key_fields: list[str] | None = None,
        threshold: float = 0.5,
) -> MatchResult:
    """Ultra-optimized Jaccard matching utilizing sparse matrix multiplication

    to eliminate Python loops entirely for similarity calculations.
    """
    num_pred = len(predicted_rows)
    num_exp = len(expected_rows)

    if num_pred == 0 or num_exp == 0:
        return {
            "matches": [],
            "pred_unmatched": list(range(num_pred)),
            "exp_unmatched": list(range(num_exp)),
        }

    if key_fields is None:
        key_fields = DEFAULT_ROW_MATCH_COLUMNS

    pred_tokens = [row_token_list(r, key_fields) for r in predicted_rows]
    exp_tokens = [row_token_list(r, key_fields) for r in expected_rows]

    vocab: dict[str, int] = {}

    # OPTIMIZATION 2: Localize dictionary setdefault references to cache inside hot loops
    def build_csr_inputs(token_lists: list[list[str]]):
        data, row_ind, col_ind = [], [], []
        set_vocab_index = vocab.setdefault

        for r_idx, t_list in enumerate(token_lists):
            for token in t_list:
                c_idx = set_vocab_index(token, len(vocab))
                data.append(1)
                row_ind.append(r_idx)
                col_ind.append(c_idx)
        return data, row_ind, col_ind

    p_data, p_rows, p_cols = build_csr_inputs(pred_tokens)
    e_data, e_rows, e_cols = build_csr_inputs(exp_tokens)

    vocab_size = len(vocab)
    if vocab_size == 0:
        vocab_size = 1

    p_matrix = sp.csr_matrix((p_data, (p_rows, p_cols)), shape=(num_pred, vocab_size), dtype=np.int32)
    e_matrix = sp.csr_matrix((e_data, (e_rows, e_cols)), shape=(num_exp, vocab_size), dtype=np.int32)

    # Intersection counts generated via high speed numerical dot products
    intersection_matrix = np.asarray(p_matrix.dot(e_matrix.T).todense())

    pred_lens = np.asarray(p_matrix.sum(axis=1)).squeeze(axis=1)
    exp_lens = np.asarray(e_matrix.sum(axis=1)).squeeze(axis=1)

    p_grid, e_grid = np.meshgrid(pred_lens, exp_lens, indexing='ij')
    union_matrix = p_grid + e_grid - intersection_matrix
    union_matrix[union_matrix == 0] = 1

    jaccard_matrix = intersection_matrix / union_matrix
    jaccard_matrix[(p_grid == 0) & (e_grid == 0)] = 1.0

    cost_matrix = np.full((num_pred, num_exp), 2.0, dtype=np.float64)

    valid_mask = jaccard_matrix >= threshold
    cost_matrix[valid_mask] = 1.0 - jaccard_matrix[valid_mask]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    # Native numerical matrix loop evaluation tracking matches ceiling costs
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.0:
            matches.append((int(r), int(c)))

    matched_pred = {r for r, _ in matches}
    matched_exp = {c for _, c in matches}

    return {
        "matches": matches,
        "pred_unmatched": [i for i in range(num_pred) if i not in matched_pred],
        "exp_unmatched": [j for j in range(num_exp) if j not in matched_exp],
    }


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