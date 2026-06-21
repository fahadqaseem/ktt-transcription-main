from __future__ import annotations

from typing import List, Dict, Tuple

from ktt_ocr_eval.core.columns import DEFAULT_ROW_MATCH_COLUMNS
from ktt_ocr_eval.evaluation.normalize import normalize_value, tokens


def composite_key(record: Dict[str, str], key_fields: List[str] | None = None) -> str:
	"""Create a deterministic composite key from a row using selected fields.

	Normalizes values before joining so small formatting differences don't
	affect exact-key matching.
	"""
	if key_fields is None:
		key_fields = DEFAULT_ROW_MATCH_COLUMNS
	parts: List[str] = []
	for field in key_fields:
		parts.append(normalize_value(record.get(field, "")))
	return "||".join(parts)


def match_by_exact_key(
	predicted_rows: List[Dict[str, str]],
	expected_rows: List[Dict[str, str]],
	key_fields: List[str] | None = None,
) -> Dict[str, List[Tuple[int, int]] | List[int]]:
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

	exp_map: Dict[str, List[int]] = {}
	for i, row in enumerate(expected_rows):
		key = composite_key(row, key_fields)
		exp_map.setdefault(key, []).append(i)

	matches: List[Tuple[int, int]] = []
	pred_unmatched: List[int] = []

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
	exp_unmatched: List[int] = []
	for indices in exp_map.values():
		exp_unmatched.extend(indices)

	return {
		"matches": matches,
		"pred_unmatched": pred_unmatched,
		"exp_unmatched": exp_unmatched,
	}


def match_by_jaccard(
	predicted_rows: List[Dict[str, str]],
	expected_rows: List[Dict[str, str]],
	key_fields: List[str] | None = None,
	threshold: float = 0.5,
) -> Dict[str, List[Tuple[int, int]] | List[int]]:
	"""Match predicted to expected rows using token Jaccard similarity on composite key fields.

	For each predicted row, find the highest-Jaccard expected row above `threshold`.
	Matches are one-to-one and expected rows are consumed when matched.
	"""
	if key_fields is None:
		key_fields = DEFAULT_ROW_MATCH_COLUMNS

	# precompute token sets for expected rows
	exp_tokens: Dict[int, set[str]] = {}
	for i, row in enumerate(expected_rows):
		combined = " ".join(tokens(row.get(f, "")) for f in key_fields)
		exp_tokens[i] = set(combined.split()) if combined else set()

	matches: List[Tuple[int, int]] = []
	pred_unmatched: List[int] = []
	matched_exp: set[int] = set()

	for pi, prow in enumerate(predicted_rows):
		combined = " ".join(tokens(prow.get(f, "")) for f in key_fields)
		pset = set(combined.split()) if combined else set()
		best_score = 0.0
		best_ei = None
		for ei, eset in exp_tokens.items():
			if ei in matched_exp:
				continue
			if not eset and not pset:
				score = 1.0
			elif not eset or not pset:
				score = 0.0
			else:
				inter = len(pset & eset)
				uni = len(pset | eset)
				score = inter / uni if uni > 0 else 0.0
			if score > best_score:
				best_score = score
				best_ei = ei
		if best_ei is not None and best_score >= threshold:
			matches.append((pi, best_ei))
			matched_exp.add(best_ei)
		else:
			pred_unmatched.append(pi)

	exp_unmatched = [i for i in range(len(expected_rows)) if i not in matched_exp]
	return {"matches": matches, "pred_unmatched": pred_unmatched, "exp_unmatched": exp_unmatched}


def match_rows(
	predicted_rows: List[Dict[str, str]],
	expected_rows: List[Dict[str, str]],
	key_fields: List[str] | None = None,
	strategy: str = "exact",
	threshold: float = 0.5,
) -> Dict[str, List[Tuple[int, int]] | List[int]]:
	if strategy == "exact":
		return match_by_exact_key(predicted_rows, expected_rows, key_fields=key_fields)
	if strategy == "jaccard":
		return match_by_jaccard(predicted_rows, expected_rows, key_fields=key_fields, threshold=threshold)
	raise ValueError(f"Unknown match strategy: {strategy}")

