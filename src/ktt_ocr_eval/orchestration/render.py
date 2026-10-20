from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Metrics where a smaller value is better; everything else ranks higher-is-better.
LOWER_IS_BETTER = {"failure_rate", "records_failed", "fp", "fn", "cer", "char_errors"}
DEFAULT_METRICS = ["precision", "recall", "f1", "failure_rate"]


def load_scores(path: Path) -> list[dict[str, Any]]:
    """Load tidy score rows from a scores jsonl file, or the latest scores-*.jsonl
    in a folder (falling back to a legacy scores.jsonl)."""
    if path.is_dir():
        candidates = sorted(path.glob("scores-*.jsonl"))
        if not candidates and (path / "scores.jsonl").exists():
            candidates = [path / "scores.jsonl"]
        if not candidates:
            raise FileNotFoundError(f"no scores-*.jsonl found in {path}")
        # Newest by write time; run-id timestamps are second-resolution, so fall
        # back to the name for a deterministic tiebreak within the same second.
        path = max(candidates, key=lambda candidate: (candidate.stat().st_mtime, candidate.name))
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _column(row: dict[str, Any]) -> str:
    field = row.get("field")
    return f"{row['metric']}:{field}" if field else row["metric"]


def _label(row: dict[str, Any]) -> str:
    nickname = row.get("nickname")
    if nickname:
        return nickname
    fingerprint = (row.get("fingerprint") or "")[:8]
    return f"{row['method']}@{fingerprint}" if fingerprint else row["method"]


def _is_lower_better(column: str) -> bool:
    return column.split(":", 1)[0] in LOWER_IS_BETTER


def fractional_ranks(values: list[float], *, lower_is_better: bool) -> list[float]:
    """Competition ranks (1 = best) where ties share the average of the positions
    they span — e.g. [0.9, 0.9, 0.8] higher-is-better -> [1.5, 1.5, 3.0]."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not lower_is_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = sum(range(i + 1, j + 2)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def build_table(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, dict[str, Any]], dict[str, float]]:
    """Pivot tidy score rows into (ordered_labels, columns, cells, avg_rank).

    Rows (variants) are ordered by mean rank across the shown metric columns,
    each column equally weighted; the rank in a column (not the value) is what
    aggregates, so metrics on different scales compare fairly.
    """
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        cells.setdefault(_label(row), {})[_column(row)] = row["value"]

    if metrics:
        columns = list(metrics)
    else:
        columns = [c for c in DEFAULT_METRICS if any(c in cell for cell in cells.values())]

    rank_sum = dict.fromkeys(cells, 0.0)
    rank_count = dict.fromkeys(cells, 0)
    for column in columns:
        scored = [(label, cell[column]) for label, cell in cells.items() if column in cell]
        if not scored:
            continue
        ranks = fractional_ranks(
            [value for _, value in scored],
            lower_is_better=_is_lower_better(column),
        )
        for (label, _), rank in zip(scored, ranks):
            rank_sum[label] += rank
            rank_count[label] += 1

    avg_rank = {
        label: (rank_sum[label] / rank_count[label]) if rank_count[label] else float("inf")
        for label in cells
    }
    ordered = sorted(cells, key=lambda label: (avg_rank[label], label))
    return ordered, columns, cells, avg_rank


def render_table(rows: list[dict[str, Any]], metrics: list[str] | None = None) -> str:
    if not rows:
        return "No scores found."
    ordered, columns, cells, avg_rank = build_table(rows, metrics)
    header = ["method", "avg_rank", *columns]
    table = [
        [label, f"{avg_rank[label]:.2f}", *[_fmt(cells[label].get(c)) for c in columns]]
        for label in ordered
    ]
    return _format_table(header, table)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        out = [cells[0].ljust(widths[0])]
        out += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  ".join(out)

    lines = [format_row(header), "  ".join("-" * width for width in widths)]
    lines += [format_row(row) for row in rows]
    return "\n".join(lines)
