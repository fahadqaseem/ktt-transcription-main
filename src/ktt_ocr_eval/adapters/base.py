from __future__ import annotations

from typing import Protocol

from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig, OCRPrediction


class OCRAdapter(Protocol):
    method: MethodConfig

    def predict(self, document: DocumentInput) -> OCRPrediction:
        """Extract page header and rows from a document."""
