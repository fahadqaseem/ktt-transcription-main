"""
ktt-ocr - glm.py

Description: This is the ZhipuAI GLM-4V adapter for the ktt-ocr-eval pipeline.
             It handles structured extraction of handwritten records from
             scanned document images.

             This adapter does two main things:
                 1. Mock/Test mode (USE_MOCK = True) --> Reads from a local fixture file
                                                    inside tests/fixtures/glm/ with no
                                                    network call and no API key needed.
                                                    This is for testing the code.
                 2. Live mode (USE_MOCK = False) --> Calls the ZhipuAI API and returns
                                                    a validated OCRPrediction.

                                                    Under the hood:
                                                    - Base64-encodes the document image
                                                    - Sends it to GLM-4V with a JSON
                                                      extraction prompt
                                                    - Parses and validates the response
                                                      with Pydantic
                                                    - Retries transient errors (rate
                                                      limits, connection resets, 5xx)
                                                      with exponential backoff
                                                    - Returns a fully assembled
                                                      OCRPrediction object

             Note: GLM-4V does not support Structured Outputs like OpenAI.
             The model is prompted to return JSON and the response is parsed
             with Pydantic. Markdown fence stripping is applied defensively.

             Registry registration (add to registry.py):
                 from ktt_ocr_eval.adapters.glm import GLMAdapter
                 ADAPTERS["glm"] = GLMAdapter

             YAML method config keys (all optional):
                 model:        ZhipuAI model name              (default: "glm-4v")
                 max_tokens:   Max completion tokens            (default: 4096)
                 max_retries:  Retry attempts on transient err  (default: 3)
                 mock_fixture: Fixture filename inside
                               tests/fixtures/glm/ to use when
                               USE_MOCK is True                 (default: "glm_mock.json")
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
from __future__ import annotations
import base64
import json
import re
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

Return ONLY a valid JSON object with this exact shape — no markdown, no explanation:

{
  "header": {
    "school": "", "grade": "", "teacher": "",
    "Date_of_Record": "", "schoolyear": "", "semester": ""
  },
  "rows": [
    {
      "lastname": "", "firstname": "", "Sex": "", "age": "",
      "address": "", "town": "", "comment": "", "days_absent": "",
      "vaccination": "", "note": "", "Date_Left": "", "DoB": ""
    }
  ]
}
"""


#========================= STRUCTURED OUTPUT SCHEMA ==================================

#These models define the expected JSON shape for Pydantic validation.
#They are intentionally separate from PredictionHeader/PredictionRecord in schemas.py,
#those are the pipeline's internal types; these are the API contract with the model.

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

#This exception is raised when the GLM adapter cannot produce a usable response.
#The project runner should catch this and log/skip the document rather than
#letting it crash the whole pipeline.
class GLMAdapterError(Exception):
    pass


#================================== ADAPTER ==========================================

#This class is the OCR adapter backed by ZhipuAI GLM-4V vision model
class GLMAdapter:

    #====== Setup ======

    def __init__(self, method: MethodConfig) -> None:
        self.method = method

        #Scalar config, using method.config.get() directly
        #(method.config_mapping() is only for nested dict values)
        self._model:        str = str(self.method.config.get("model", "glm-4v"))
        self._max_tokens:   int = int(self.method.config.get("max_tokens", 4096))
        self._max_retries:  int = int(self.method.config.get("max_retries", 3))
        self._mock_fixture: str = str(self.method.config.get("mock_fixture", "glm_mock.json"))

        #Only importing and authenticating when we actually need the live client
        if not USE_MOCK:
            import os
            from zhipuai import ZhipuAI

            api_key = os.environ.get("ZHIPUAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ZHIPUAI_API_KEY environment variable is not set."
                )
            self._client = ZhipuAI(api_key = api_key)


    #====== Public ======

    def predict(self, document: DocumentInput) -> OCRPrediction:
        """
        This method runs extraction on the document and returns a validated OCRPrediction.

        When USE_MOCK is True: reads from the local fixture, no API call is made.
        When USE_MOCK is False: calls the ZhipuAI API.
        Both paths produce the same (header_dict, rows_list, usage_dict) tuple
        so the OCRPrediction assembly below is identical in both modes.
        """
        if USE_MOCK:
            mock_choice = self._get_mock_response()

            if mock_choice["finish_reason"] == "length":
                raise GLMAdapterError(
                    f"Response was truncated (finish_reason='length'). "
                    f"Raise max_tokens above the current {self._max_tokens}."
                )
            raw_content = mock_choice["content"]
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
                "provider":          "glm",
                "model":             self._model,
                "mock":              USE_MOCK,
                "usage":             usage,
                "document_metadata": document.metadata,
            },
            "metadata": {
                "adapter": "glm",
                "model":   self._model,
                "mock":    USE_MOCK,
            },
        })


    #====== Private ======

    def _call_api(
        self, document: DocumentInput
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, int]]:
        """
        This method sends the document image to ZhipuAI and returns
        (raw_content, header_dict, rows_list, usage_dict).

        GLM-4V does not support Structured Outputs, so the model is prompted
        to return JSON directly. The response is validated with Pydantic.

        Retries on transient errors (rate limits, connection resets, 5xx) with
        exponential backoff. Non-retryable errors raise GLMAdapterError immediately.
        """
        from zhipuai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
        )

        b64, media_type = _encode_image(document.source_path)

        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model = self._model,
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{b64}",
                                    },
                                },
                                {"type": "text", "text": _EXTRACTION_PROMPT},
                            ],
                        }
                    ],
                    max_tokens = self._max_tokens,
                    temperature = 0,
                )
                break

            except (APIConnectionError, APITimeoutError) as exc:
                #Always retry connection and timeout errors
                last_exc = exc
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt #1s, 2s, 4s...
                    time.sleep(wait)
            except APIStatusError as exc:
                #Retry rate limits (429) and server errors (5xx), fail fast on others
                if exc.status_code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    if attempt < self._max_retries - 1:
                        wait = 2 ** attempt
                        time.sleep(wait)
                else:
                    raise GLMAdapterError(
                        f"ZhipuAI API error {exc.status_code}: {exc}"
                    ) from exc
            except Exception as exc:
                raise GLMAdapterError(f"ZhipuAI API call failed: {exc}") from exc
        else:
            raise GLMAdapterError(
                f"ZhipuAI API call failed after {self._max_retries} attempts. "
                f"Last error: {last_exc}"
            ) from last_exc

        choice = response.choices[0]

        if choice.finish_reason == "length":
            raise GLMAdapterError(
                f"Response was truncated (finish_reason='length'). "
                f"Raise max_tokens above the current {self._max_tokens}."
            )

        raw_content = choice.message.content
        if not raw_content:
            raise GLMAdapterError("ZhipuAI returned an empty response content.")

        header, rows = _parse_content(raw_content)

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }

        return raw_content, header, rows, usage


    def _get_mock_response(self) -> dict[str, str]:
        """
        This method reads a saved ZhipuAI API response from the fixture file and
        returns a minimal choice-like dict: {"content": "<json string>", "finish_reason": "stop"}

        Returning both fields (not just content) means predict() can run the same
        finish_reason check it runs on live responses, so the truncated fixture
        actually exercises the GLMAdapterError("length") branch.

        The fixture file is controlled by the "mock_fixture" method config key
        (default: "glm_mock.json"), so different scenarios can be tested from
        YAML without touching this file:
            config:
              mock_fixture: glm_mock_truncated.json

        Note: usage is still not covered by mock mode, use
        unittest.mock to patch self._client for that branch.

        Expected fixture shape:
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{...json...}"},
                        "finish_reason": "stop"
                    }
                ]
            }
        """
        fixture_path = _FIXTURES_DIR / self._mock_fixture

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"GLM mock fixture not found at: {fixture_path}\n"
                "Create the file or set USE_MOCK = False to use the live API."
            )

        try:
            fixture: dict[str, Any] = json.loads(
                fixture_path.read_text(encoding = "utf-8")
            )
            choice = fixture["choices"][0]
            return {
                "content":       choice["message"]["content"],
                "finish_reason": choice.get("finish_reason", "stop"),
            }

        except (KeyError, IndexError) as exc:
            raise GLMAdapterError(
                f"Fixture has unexpected shape: {exc}. "
                'Expected {"choices": [{"message": {"content": "..."}, "finish_reason": "..."}]}'
            ) from exc


#========================== MODULE-LEVEL HELPERS =====================================

#This maps common image file extensions to their MIME types for the API payload
_MEDIA_TYPES: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
}


def _encode_image(path: Path | str) -> tuple[str, str]:
    """
    This function encodes an image file as base64 and returns (b64_string, media_type).
    NOTE: historical TIFF scans can be very large, if token cost becomes an issue,
    consider resizing / converting to PNG before encoding.
    """
    path = Path(path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return b64, media_type


def _parse_content(
    raw: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    This function parses the model's response string into (header_dict, rows_list).

    Uses Pydantic (_ExtractionPayload) for validation. Since GLM-4V does not
    support Structured Outputs, the model may occasionally wrap its response in
    markdown fences despite being instructed not to — fence stripping is applied
    defensively before parsing.

    If the JSON is malformed or doesn't match the schema, Pydantic's
    ValidationError is caught and re-raised as GLMAdapterError so callers only
    ever need to catch one exception type per provider. The original field-level
    message is preserved via `from exc`.
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
        raise GLMAdapterError(f"Invalid structured output: {exc}") from exc

    return payload.header.model_dump(), [r.model_dump() for r in payload.rows]
