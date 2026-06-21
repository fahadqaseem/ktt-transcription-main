from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ktt_ocr_eval.adapters import create_adapter
from ktt_ocr_eval.core.artifacts import append_prediction
from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig, OCRPrediction


def load_manifest(path: Path) -> list[DocumentInput]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"document_id", "source_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    documents: list[DocumentInput] = []
    for record in frame.to_dict(orient="records"):
        metadata = {key: value for key, value in record.items() if key not in required}
        documents.append(
            DocumentInput(
                document_id=record["document_id"],
                source_path=Path(record["source_path"]),
                metadata=metadata,
            )
        )
    return documents


def run_matrix(
    documents: Iterable[DocumentInput],
    methods: Iterable[MethodConfig],
    out_dir: Path,
    *,
    only_method: str | None = None,
    resume: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    completed = load_completed_keys(predictions_path) if resume else set()
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for method in methods:
        if only_method and method.name != only_method:
            continue
        adapter_error: Exception | None = None
        try:
            adapter = create_adapter(method)
        except Exception as exc:  # noqa: BLE001 - capture setup failures as artifacts.
            adapter = None
            adapter_error = exc

        for document in documents:
            key = (method.name, document.document_id)
            if key in completed:
                continue
            try:
                if adapter_error:
                    raise adapter_error
                if adapter is None:
                    raise RuntimeError("adapter was not initialized")
                prediction = adapter.predict(document)
            except Exception as exc:  # noqa: BLE001 - capture method failures as artifacts.
                prediction = OCRPrediction(
                    document_id=document.document_id,
                    method=method.name,
                    source_path=str(document.source_path),
                    metadata={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            append_prediction(predictions_path, prediction)
            completed.add(key)

    write_run_metadata(out_dir, methods, run_started)
    return predictions_path


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            completed.add((payload["method"], payload["document_id"]))
    return completed


def write_run_metadata(out_dir: Path, methods: Iterable[MethodConfig], started_at: str) -> None:
    payload = {
        "started_at": started_at,
        "methods": [method.model_dump(mode="json") for method in methods],
    }
    (out_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
