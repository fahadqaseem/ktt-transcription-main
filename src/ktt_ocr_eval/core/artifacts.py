from __future__ import annotations
import pandas as pd
import json
from collections.abc import Iterable
from pathlib import Path

from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_ocr_eval.core.schemas import OCRPrediction, PredictionHeader, PredictionRecord

def write_predictions(path: Path, predictions: Iterable[OCRPrediction]) -> None:
    """Writes JSON Lines predictions using a heavy I/O buffer to save disk overhead."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # 64KB operational buffer handles physical disk flushes in clean sequences
    with path.open("w", encoding="utf-8", buffering=64 * 1024) as file:
        # Calling file.write inline directly via mapping is faster than manual for-loops
        for prediction in predictions:
            file.write(prediction.model_dump_json(exclude_none=True))
            file.write("\n")


def append_prediction(path: Path, prediction: OCRPrediction) -> None:
    """Appends a single prediction artifact to the destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(prediction.model_dump_json(exclude_none=True))
        file.write("\n")


def read_predictions(path: Path) -> list[OCRPrediction]:
    """Highly optimized stream deserializer converting JSON strings via C-level iterators."""
    validate = OCRPrediction.model_validate
    parse_json = json.loads

    # OPTIMIZATION: Cache list append method directly as a local variable.
    # Looking up a method name on an object inside a hot loop is an O(1) penalty
    # we can remove by pinning the raw bound function to a stack address variable.
    predictions: list[OCRPrediction] = []
    push_to_list = predictions.append

    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                # Executes via our high speed pinned function references
                push_to_list(validate(parse_json(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc

    return predictions


def read_predictions_from_csv(path: Path) -> list[OCRPrediction]:
    """Ingests a flattened OCR prediction CSV and structures it for evaluation."""
    # Read everything cleanly as strings to prevent automatic number formatting drops
    df = pd.read_csv(path, dtype=str).fillna("")
    predictions: list[OCRPrediction] = []

    # Speed optimization: Group rows by document_id and method simultaneously
    for (doc_id, method), group in df.groupby(["document_id", "method"], sort=False):
        # 1. Grab page headers from the very first row of this document block
        first_row = group.iloc[0]
        header = PredictionHeader(
            school=first_row.get("school"),
            grade=first_row.get("grade"),
            teacher=first_row.get("teacher"),
            Date_of_Record=first_row.get("Date_of_Record"),
            schoolyear=first_row.get("schoolyear"),
            semester=first_row.get("semester"),
        )

        # 2. Extract table rows for this document
        rows: list[PredictionRecord] = []
        # Convert records directly to an underlying dictionary stream for maximum speed
        for record in group.to_dict(orient="records"):
            rows.append(PredictionRecord(
                lastname=record.get("lastname"),
                firstname=record.get("firstname"),
                Sex=record.get("Sex"),
                age=record.get("age"),
                address=record.get("address"),
                town=record.get("town"),
                comment=record.get("comment"),
                days_absent=record.get("days_absent"),
                vaccination=record.get("vaccination"),
                note=record.get("note"),
                Date_Left=record.get("Date_Left"),
                DoB=record.get("DoB"),
            ))

        # 3. Rebuild the canonical OCRPrediction object
        predictions.append(OCRPrediction(
            document_id=str(doc_id),
            method=str(method),
            header=header,
            rows=rows
        ))

    return predictions