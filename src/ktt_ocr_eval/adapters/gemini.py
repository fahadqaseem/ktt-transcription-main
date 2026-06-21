"""
ktt-ocr - gemini.py

Description: This is the Google Gemini adapter for the ktt-ocr-eval pipeline.
             It handles structured extraction of handwritten records from
             scanned document images.

             This adapter does two main things:
                 1. Mock/Test mode (USE_MOCK = True) --> Reads from a local fixture file
                                                    inside tests/fixtures/ with no
                                                    network call and no API key needed.
                                                    This is for testing the code.
                 2. Live mode (USE_MOCK = False) --> Calls the Gemini API with a JSON
                                                    response schema and returns
                                                    a validated OCRPrediction.

                                                    Under the hood:
                                                    - Loads the document image with PIL
                                                    - Sends it to Gemini with a typed
                                                      schema (_ExtractionPayload) via
                                                      response_schema
                                                    - Validates the response with Pydantic
                                                    - Retries transient errors (rate
                                                      limits, timeouts, 5xx) with
                                                      exponential backoff
                                                    - Returns a fully assembled
                                                      OCRPrediction object

             Registry registration (add to registry.py):
                 from ktt_ocr_eval.adapters.gemini import GeminiAdapter
                 ADAPTERS["gemini"] = GeminiAdapter

             YAML method config keys (all optional):
                 model:        Gemini model name               (default: "gemini-2.0-flash")
                 max_tokens:   Max output tokens                (default: 4096)
                 max_retries:  Retry attempts on transient err  (default: 3)
                 mock_fixture: Fixture filename inside
                               tests/fixtures/ to use when
                               USE_MOCK is True                 (default: "gemini_mock.json")
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

#The third-party imports:
from pydantic import BaseModel, ValidationError
from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig, OCRPrediction


#============================== MOCK TOGGLE ==========================================

#Flip this to False when you're ready to hit the real API.
#When True, no network calls are made and no API key is needed,
#the adapter reads from tests/fixtures instead.
USE_MOCK: bool = True   # <-- flip to False when you're ready to hit the real API


#============================== FIXTURE PATH =========================================

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests" / "fixtures"
)


#============================== EXTRACTION PROMPT ====================================

_EXTRACTION_PROMPT: str = """\
You are extracting structured data from a historical school enrollment record
(circa 1880–1930).

The image is a scanned handwritten table that may contain:
- faded ink
- handwriting variations
- abbreviations
- damaged paper
- partially illegible text

Instructions:

1. Extract every visible student row.
2. Extract page-level header information when present.
3. Preserve original spelling exactly as written.
4. Preserve abbreviations exactly as written.
5. Do not expand abbreviations.
6. Do not normalize names, locations, or dates.
7. Do not infer missing information.
8. If a field is blank, missing, or illegible, return "".
9. If only part of a value is legible, return only the legible portion.
10. Extract only information visible in the image.
11. Do not invent students, rows, or values.
12. Maintain the original row order.
"""


#========================= STRUCTURED OUTPUT SCHEMA ==================================

#These models define the exact JSON shape Gemini will be constrained to via
#response_schema. They are intentionally separate from PredictionHeader/
#PredictionRecord in schemas.py, those are the pipeline's internal types,
#these are the API contract with the model.

class _Header(BaseModel):
    school:         str = ""
    grade:          str = ""
    teacher:        str = ""
    Date_of_Record: str = ""
    schoolyear:     str = ""
    semester:       str = ""


class _Row(BaseModel):
    lastname:    str = ""
    firstname:   str = ""
    Sex:         str = ""
    age:         str = ""
    address:     str = ""
    town:        str = ""
    comment:     str = ""
    days_absent: str = ""
    vaccination: str = ""
    note:        str = ""
    Date_Left:   str = ""
    DoB:         str = ""


class _ExtractionPayload(BaseModel):
    header: _Header
    rows:   list[_Row]


#============================== CUSTOM EXCEPTION =====================================

#This exception is raised when the Gemini adapter cannot produce a usable response.
#The project runner should catch this and log/skip the document rather than
#letting it crash the whole pipeline.
class GeminiAdapterError(Exception):
    pass


#================================== ADAPTER ==========================================

#This class is the OCR adapter backed by Google Gemini vision with a JSON response schema
class GeminiAdapter:

    #====== Setup ======

    def __init__(self, method: MethodConfig) -> None:
        self.method = method

        #Scalar config, using method.config.get() directly
        #(method.config_mapping() is only for nested dict values)
        self._model:        str = str(self.method.config.get("model", "gemini-2.0-flash"))
        self._max_tokens:   int = int(self.method.config.get("max_tokens", 4096))
        self._max_retries:  int = int(self.method.config.get("max_retries", 3))
        self._mock_fixture: str = str(self.method.config.get("mock_fixture", "gemini_mock.json"))

        #Only importing and authenticating when we actually need the live client
        if not USE_MOCK:
            import os
            import google.generativeai as genai

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY environment variable is not set."
                )
            genai.configure(api_key = api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(self._model)


    #====== Public ======

    def predict(self, document: DocumentInput) -> OCRPrediction:
        """
        This method runs extraction on the document and returns a validated OCRPrediction.

        When USE_MOCK is True: reads from the local fixture, no API call is made.
        When USE_MOCK is False: calls the Gemini API with a JSON response schema.
        Both paths produce the same (header_dict, rows_list, usage_dict) tuple
        so the OCRPrediction assembly below is identical in both modes.
        """
        if USE_MOCK:
            mock_candidate = self._get_mock_response()

            #Running the same finish_reason check as the live path so the
            #truncated / safety / recitation fixtures actually exercise
            #those error branches.
            _check_finish_reason(mock_candidate["finish_reason"], self._max_tokens)

            raw_content = mock_candidate["content"]
            header, rows = _parse_content(raw_content)
            usage: dict[str, int] = {}
        else:
            raw_content, header, rows, usage = self._call_api(document)

        return OCRPrediction.model_validate({
            "document_id": document.document_id,
            "method":      self.method.name,
            "source_path": str(document.source_path),
            "header":      header,
            "rows":        rows,
            "raw_text":    raw_content,
            "raw_output": {
                "provider":          "gemini",
                "model":             self._model,
                "mock":              USE_MOCK,
                "usage":             usage,
                "document_metadata": document.metadata,
            },
            "metadata": {
                "adapter": "gemini",
                "model":   self._model,
                "mock":    USE_MOCK,
            },
        })


    #====== Private ======

    def _call_api(
        self, document: DocumentInput
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, int]]:
        """
        This method sends the document image to Gemini using a JSON response schema and
        returns (raw_content, header_dict, rows_list, usage_dict).

        response_mime_type = "application/json" combined with response_schema is
        Gemini's equivalent of OpenAI's Structured Outputs, it constrains the
        model's output to match _ExtractionPayload's shape.

        Retries on transient errors (429 ResourceExhausted, 503 ServiceUnavailable,
        DeadlineExceeded) with exponential backoff. Non-retryable errors raise
        GeminiAdapterError immediately.
        """
        from google.api_core.exceptions import (
            DeadlineExceeded,
            ResourceExhausted,
            ServiceUnavailable,
        )
        import PIL.Image

        image = PIL.Image.open(str(document.source_path))

        generation_config = self._genai.types.GenerationConfig(
            temperature = 0.0,
            max_output_tokens = self._max_tokens,
            response_mime_type = "application/json",
            response_schema = _ExtractionPayload,
        )

        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.generate_content(
                    [_EXTRACTION_PROMPT, image],
                    generation_config = generation_config,
                )
                break

            except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt #1s, 2s, 4s...
                    time.sleep(wait)
            except Exception as exc:
                raise GeminiAdapterError(f"Gemini API call failed: {exc}") from exc
        else:
            raise GeminiAdapterError(
                f"Gemini API call failed after {self._max_retries} attempts. "
                f"Last error: {last_exc}"
            ) from last_exc

        if not response.candidates:
            raise GeminiAdapterError("Gemini returned no candidates.")

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name #enum -> "STOP", "MAX_TOKENS", etc.

        _check_finish_reason(finish_reason, self._max_tokens)

        raw_content = response.text

        try:
            payload = _ExtractionPayload.model_validate_json(raw_content)
        except ValidationError as exc:
            raise GeminiAdapterError(f"Invalid structured output: {exc}") from exc

        header = payload.header.model_dump()
        rows   = [r.model_dump() for r in payload.rows]

        usage: dict[str, int] = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens":     response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens":      response.usage_metadata.total_token_count,
            }

        return raw_content, header, rows, usage


    def _get_mock_response(self) -> dict[str, str]:
        """
        This method reads a saved Gemini API response from the fixture file and returns
        a minimal candidate-like dict: {"content": "<json string>", "finish_reason": "STOP"}

        Returning both fields (not just content) means predict() can run the same
        finish_reason check it runs on live responses, so the truncated / safety /
        recitation fixtures actually exercise those error branches.

        The fixture file is controlled by the "mock_fixture" method config key
        (default: "gemini_mock.json"), so different scenarios can be tested from
        YAML without touching this file:
            config:
              mock_fixture: gemini_mock_truncated.json

        Note: usage_metadata is still not covered by mock mode, use
        unittest.mock to patch self._client for that.

        Expected fixture shape:
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "{...json...}"}]},
                        "finish_reason": "STOP"
                    }
                ]
            }
        """
        fixture_path = _FIXTURES_DIR / self._mock_fixture

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Gemini mock fixture not found at: {fixture_path}\n"
                "Create the file or set USE_MOCK = False to use the live API."
            )

        try:
            fixture: dict[str, Any] = json.loads(
                fixture_path.read_text(encoding = "utf-8")
            )
            candidate = fixture["candidates"][0]
            parts = candidate["content"]["parts"]

            #SAFETY/RECITATION blocks often come back with an empty parts list,
            #there's no text to extract, but finish_reason is what predict()
            #needs to raise the right error.
            content = parts[0]["text"] if parts else ""

            return {
                "content":       content,
                "finish_reason": candidate.get("finish_reason", "STOP"),
            }

        except (KeyError, IndexError) as exc:
            raise GeminiAdapterError(
                f"Fixture has unexpected shape: {exc}. "
                'Expected {"candidates": [{"content": {"parts": [{"text": "..."}]}, '
                '"finish_reason": "..."}]}'
            ) from exc


#========================== MODULE-LEVEL HELPERS =====================================

def _check_finish_reason(finish_reason: str, max_tokens: int) -> None:
    """
    This function raises GeminiAdapterError for non-success finish reasons.

    Gemini finish_reason values (relevant ones):
        STOP        - normal completion
        MAX_TOKENS  - output was truncated, equivalent to OpenAI's "length"
        SAFETY      - blocked by safety filters, equivalent to OpenAI's refusal
        RECITATION  - blocked due to suspected copyrighted recitation
        OTHER       - unspecified other reason
    """
    if finish_reason == "STOP":
        return

    if finish_reason == "MAX_TOKENS":
        raise GeminiAdapterError(
            f"Response was truncated (finish_reason='MAX_TOKENS'). "
            f"Raise max_tokens above the current {max_tokens}."
        )

    if finish_reason == "SAFETY":
        raise GeminiAdapterError(
            "Gemini blocked the response due to safety filters "
            "(finish_reason='SAFETY')."
        )

    if finish_reason == "RECITATION":
        raise GeminiAdapterError(
            "Gemini blocked the response due to suspected recitation "
            "(finish_reason='RECITATION')."
        )

    raise GeminiAdapterError(f"Unexpected finish_reason: {finish_reason!r}")


def _parse_content(
    raw: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    This function parses the fixture content string into (header_dict, rows_list).

    Uses Pydantic (_ExtractionPayload) for validation rather than bare json.loads,
    meaning the mock path validates the same schema the live response_schema enforces.

    If the fixture JSON is malformed or doesn't match the schema, Pydantic's
    ValidationError is caught and re-raised as GeminiAdapterError so callers only
    ever need to catch one exception type per provider, the original field-level
    message is preserved via `from exc`. Returning {}, [] here would just cause
    a confusing "missing field" crash further downstream.

    Only strips markdown fences defensively, no greedy regex recovery.
    """
    text = raw.strip()
    if text.startswith("```"):
        #The model ignored instructions and wrapped output in fences anyway
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        payload = _ExtractionPayload.model_validate_json(text)
    except ValidationError as exc:
        raise GeminiAdapterError(f"Invalid structured output: {exc}") from exc

    return payload.header.model_dump(), [r.model_dump() for r in payload.rows]
