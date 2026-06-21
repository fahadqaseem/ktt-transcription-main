from __future__ import annotations

from ktt_ocr_eval.core.schemas import (
    DocumentInput,
    MethodConfig,
    OCRPrediction,
)


class ExampleProviderAdapter:
    """Minimal OCR provider example.

    Copy this file when adding a new provider. The only hard requirement is that
    `predict` returns an `OCRPrediction` with a page-level header and row list.

    This example reads header/row data from method config so it can run without
    API keys, image files, or model installs.
    """

    def __init__(self, method: MethodConfig):
        self.method = method
        self.default_header = method.config_mapping("default_header")
        self.example_rows = method.config_sequence("example_rows")

    def predict(self, document: DocumentInput) -> OCRPrediction:
        return OCRPrediction.model_validate(
            {
                "document_id": document.document_id,
                "method": self.method.name,
                "source_path": str(document.source_path),
                "header": self.default_header,
                "rows": self.example_rows,
                "raw_text": "Example provider output. Replace this with OCR/model text.",
                "raw_output": {
                    "provider": "example_provider",
                    "document_metadata": document.metadata,
                },
                "metadata": {
                    "adapter": "example_provider",
                    "notes": "Template adapter for new OCR providers.",
                },
            }
        )
