"""
ktt-ocr - textract.py

Description: This is the AWS Textract adapter for the ktt-ocr-eval pipeline.
             It handles structured extraction of handwritten records from
             scanned document images using Textract's table analysis.

             Unlike the LLM adapters (GPT, Gemini, GLM), Textract returns a
             flat list of typed Block objects connected by a relationship graph
             — not text content. This adapter traverses that graph to
             reconstruct table structure, then maps cells to schema fields.

             This adapter does two main things:
                 1. Mock/Test mode (USE_MOCK = True) --> Reads from a local fixture file
                                                    inside tests/fixtures/textract/ with
                                                    no AWS credentials needed.
                                                    This is for testing the code.
                 2. Live mode (USE_MOCK = False) --> Calls AWS Textract AnalyzeDocument
                                                    with TABLE + FORM feature types and
                                                    returns a validated OCRPrediction.

                                                    Under the hood:
                                                    - Reads the document image as bytes
                                                    - Calls analyze_document via boto3
                                                    - Indexes all returned Blocks by Id
                                                    - Traverses TABLE → CELL → WORD
                                                      relationships to build 2D grids
                                                    - Maps grids to schema fields using
                                                      a configurable column index map
                                                    - Retries throttling and 5xx errors
                                                      with exponential backoff
                                                    - Returns a fully assembled
                                                      OCRPrediction object

             Column mapping:
                 The Calumet Public Schools documents have 20+ columns, but only
                 a handful map to our schema fields. The column_map config key
                 controls which 0-based column index maps to which field.
                 Defaults are set for the known Calumet document layout.
                 Override per-method in YAML if a batch has different column order.

                 The "name" field is special: it maps to a single cell containing
                 "Lastname Firstname" which is split on the first space.

             Registry registration (add to registry.py):
                 from ktt_ocr_eval.adapters.textract import TextractAdapter
                 ADAPTERS["textract"] = TextractAdapter

             YAML method config keys (all optional):
                 max_retries:  Retry attempts on transient err  (default: 3)
                 mock_fixture: Fixture filename inside
                               tests/fixtures/textract/         (default: "textract_mock.json")
                 column_map:   Dict of field → 0-based column
                               index in the student table        (default: see below)

             Default column_map (for real Calumet documents):
                 name: 0        # "Lastname Firstname" → split to lastname/firstname
                 age: 1
                 DoB: 2
                 days_absent: 4  # index 3 = days_present (not in schema)
                 vaccination: 18
                 Date_Left: 19
                 address: 20
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

#The third-party imports:
from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig, OCRPrediction


#============================== MOCK TOGGLE ==========================================

#Flip this to False when you're ready to hit the real API.
#When True, no AWS credentials are needed and boto3 does not need to be configured.
USE_MOCK: bool = True   # <-- flip to False when you're ready to hit the real API


#============================== FIXTURE PATH =========================================

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests" / "fixtures"
)


#============================== DEFAULT COLUMN MAP ===================================

#0-based column indices for the real Calumet Public Schools document layout.
#Override via config_mapping("column_map") in the YAML method config if needed.
_DEFAULT_COLUMN_MAP: dict[str, int] = {
    "name":        0,   # "Lastname Firstname" → split to lastname/firstname
    "age":         1,
    "DoB":         2,
    "days_absent": 4,   # index 3 = days_present (not in schema, skipped)
    "vaccination": 18,
    "Date_Left":   19,
    "address":     20,
}


#============================== CUSTOM EXCEPTION =====================================

#This exception is raised when the Textract adapter cannot produce a usable response.
#The project runner should catch this and log/skip the document rather than
#letting it crash the whole pipeline.
class TextractAdapterError(Exception):
    pass


#================================== ADAPTER ==========================================

#This class is the OCR adapter backed by AWS Textract AnalyzeDocument
class TextractAdapter:

    #====== Setup ======

    def __init__(self, method: MethodConfig) -> None:
        self.method = method

        #Scalar config, using method.config.get() directly
        #(method.config_mapping() is only for nested dict values)
        self._max_retries:  int = int(self.method.config.get("max_retries", 3))
        self._mock_fixture: str = str(self.method.config.get("mock_fixture", "textract_mock.json"))

        #Column map: field_name → 0-based column index in the student table.
        #config_mapping() is the right accessor here since column_map is a nested dict.
        raw_col_map = self.method.config_mapping("column_map") or _DEFAULT_COLUMN_MAP
        self._col_map: dict[str, int] = {k: int(v) for k, v in raw_col_map.items()}

        #Only importing and configuring boto3 when we actually need the live client
        if not USE_MOCK:
            import os
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            self._client = boto3.client("textract", region_name = region)


    #====== Public ======

    def predict(self, document: DocumentInput) -> OCRPrediction:
        """
        This method runs extraction on the document and returns a validated OCRPrediction.

        When USE_MOCK is True: reads from the local fixture, no AWS call is made.
        When USE_MOCK is False: calls AWS Textract AnalyzeDocument.
        Both paths return the same raw Textract response dict, so the entire
        block-graph parsing pipeline runs identically in both modes.
        """
        if USE_MOCK:
            response = self._get_mock_response()
        else:
            response = self._call_api(document)

        if "Blocks" not in response:
            raise TextractAdapterError(
                "Textract response is missing the 'Blocks' key. "
                "The response may be malformed or from an unexpected API version."
            )

        blocks    = response["Blocks"]
        block_map = {b["Id"]: b for b in blocks}

        raw_text = "\n".join(
            b["Text"]
            for b in blocks
            if b["BlockType"] == "LINE" and "Text" in b
        )

        tables        = _extract_tables(blocks, block_map)
        header, rows  = self._map_schema(tables)

        return OCRPrediction.model_validate({
            "document_id": document.document_id,
            "method":      self.method.name,
            "source_path": str(document.source_path),
            "header":      header,
            "rows":        rows,
            "raw_text":    raw_text,
            "raw_output": {
                "provider":          "aws_textract",
                "tables_detected":   len(tables),
                "blocks_total":      len(blocks),
                "mock":              USE_MOCK,
                "document_metadata": document.metadata,
            },
            "metadata": {
                "adapter":         "textract",
                "tables_detected": len(tables),
                "rows_extracted":  len(rows),
                "mock":            USE_MOCK,
            },
        })


    #====== Private ======

    def _call_api(self, document: DocumentInput) -> dict[str, Any]:
        """
        This method calls AWS Textract AnalyzeDocument and returns the raw response dict.

        Uses TABLE and FORMS feature types. TABLE gives us the 2D cell structure
        for student records. FORMS captures key-value pairs which may help with
        the pre-printed header fields.

        Retries on throttling (ThrottlingException, ProvisionedThroughputExceededException)
        and transient server errors with exponential backoff.
        Non-retryable errors raise TextractAdapterError immediately.
        """
        from botocore.exceptions import ClientError

        image_bytes = Path(document.source_path).read_bytes()

        _RETRYABLE_CODES = {
            "ThrottlingException",
            "ProvisionedThroughputExceededException",
            "InternalServerError",
            "ServiceUnavailableException",
        }

        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.analyze_document(
                    Document      = {"Bytes": image_bytes},
                    FeatureTypes  = ["TABLES", "FORMS"],
                )
                return response

            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in _RETRYABLE_CODES:
                    last_exc = exc
                    if attempt < self._max_retries - 1:
                        wait = 2 ** attempt  #1s, 2s, 4s...
                        time.sleep(wait)
                else:
                    raise TextractAdapterError(
                        f"AWS Textract error ({code}): {exc}"
                    ) from exc
            except Exception as exc:
                raise TextractAdapterError(
                    f"AWS Textract call failed: {exc}"
                ) from exc

        raise TextractAdapterError(
            f"AWS Textract call failed after {self._max_retries} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc


    def _get_mock_response(self) -> dict[str, Any]:
        """
        This method reads a saved Textract response from the fixture file and returns
        the full response dict — the same type that _call_api() returns.

        Both paths return the same dict, so the entire block-graph parsing
        pipeline in predict() runs identically in mock and live modes.

        The fixture file is controlled by the "mock_fixture" method config key
        (default: "textract_mock.json"), so different scenarios can be tested
        from YAML without touching this file:
            config:
              mock_fixture: textract_mock_single_table.json

        Expected fixture shape:
            {
                "Blocks": [
                    {"BlockType": "TABLE", "Id": "...", "Relationships": [...]},
                    {"BlockType": "CELL",  "Id": "...", "RowIndex": 1, "ColumnIndex": 1, ...},
                    {"BlockType": "WORD",  "Id": "...", "Text": "..."},
                    {"BlockType": "LINE",  "Id": "...", "Text": "..."},
                    ...
                ],
                "DocumentMetadata": {"Pages": 1}
            }
        """
        fixture_path = _FIXTURES_DIR / self._mock_fixture

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Textract mock fixture not found at: {fixture_path}\n"
                "Create the file or set USE_MOCK = False to use the live API."
            )

        try:
            return json.loads(fixture_path.read_text(encoding = "utf-8"))
        except json.JSONDecodeError as exc:
            raise TextractAdapterError(
                f"Fixture is not valid JSON: {exc}"
            ) from exc


    def _map_schema(
        self,
        tables: list[list[list[str]]],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        This method maps extracted 2D table grids to the OCRPrediction header and rows.

        Convention:
        - 2+ tables: first table = page header block, second = student data table.
        - 1 table:   treated as student data table; header fields left empty.
        - 0 tables:  both header and rows are empty (valid — Textract found no tables).

        The student table is mapped using self._col_map (field → 0-based column index).
        The name field is split on the first space into lastname / firstname.
        The first row of the student table is skipped if it looks like column labels.
        """
        header: dict[str, str] = {}
        rows:   list[dict[str, str]] = []

        if not tables:
            return header, rows

        if len(tables) >= 2:
            header_grid, student_grid = tables[0], tables[1]
        else:
            header_grid, student_grid = [], tables[0]

        #Map header table: flatten all cells and assign to header fields positionally
        _HEADER_FIELDS = ["school", "grade", "teacher", "Date_of_Record", "schoolyear", "semester"]
        flat_header = [cell for row in header_grid for cell in row if cell]
        for i, field in enumerate(_HEADER_FIELDS):
            header[field] = flat_header[i] if i < len(flat_header) else ""

        #Map student table rows using column index map
        start = 1 if student_grid and _is_label_row(student_grid[0]) else 0
        for raw_row in student_grid[start:]:
            record = self._map_row(raw_row)
            if any(record.values()):     #skip fully-empty rows
                rows.append(record)

        return header, rows


    def _map_row(self, raw_row: list[str]) -> dict[str, str]:
        """
        This method maps a single student table row to a schema record dict using
        self._col_map.  The "name" entry in col_map is split on the first space
        to produce lastname and firstname separately.
        """
        def get(idx: int) -> str:
            return raw_row[idx].strip() if idx < len(raw_row) else ""

        lastname  = ""
        firstname = ""
        if "name" in self._col_map:
            full_name = get(self._col_map["name"])
            parts = full_name.split(" ", 1)
            lastname  = parts[0].strip()
            firstname = parts[1].strip() if len(parts) == 2 else ""

        return {
            "lastname":    lastname,
            "firstname":   firstname,
            "Sex":         "",
            "age":         get(self._col_map.get("age",         -1)),
            "address":     get(self._col_map.get("address",     -1)),
            "town":        "",
            "comment":     "",
            "days_absent": get(self._col_map.get("days_absent", -1)),
            "vaccination": get(self._col_map.get("vaccination", -1)),
            "note":        "",
            "Date_Left":   get(self._col_map.get("Date_Left",   -1)),
            "DoB":         get(self._col_map.get("DoB",         -1)),
        }


#========================== MODULE-LEVEL HELPERS =====================================

def _extract_tables(
    blocks:    list[dict[str, Any]],
    block_map: dict[str, dict[str, Any]],
) -> list[list[list[str]]]:
    """
    This function traverses the Textract block relationship graph and returns a
    list of 2D tables. Each table is a list of rows; each row is a list of
    cell text strings (0-indexed, empty string for blank cells).

    Graph traversal: TABLE → CELL (via CHILD relationships) → WORD (via CHILD).
    WORD text is joined with spaces to produce the cell string.
    Merged cells (RowSpan/ColumnSpan > 1) are written to their top-left position.

    Textract RowIndex and ColumnIndex are 1-based; we convert to 0-based.
    """
    tables: list[list[list[str]]] = []

    for block in blocks:
        if block["BlockType"] != "TABLE":
            continue

        cells: dict[tuple[int, int], str] = {}
        max_row = max_col = 0

        for rel in block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue

            for cell_id in rel["Ids"]:
                cell = block_map.get(cell_id, {})
                if cell.get("BlockType") != "CELL":
                    continue

                r = cell["RowIndex"]    - 1    # → 0-based
                c = cell["ColumnIndex"] - 1
                max_row = max(max_row, r)
                max_col = max(max_col, c)

                #Collect WORD children of this cell
                words: list[str] = []
                for cell_rel in cell.get("Relationships", []):
                    if cell_rel["Type"] != "CHILD":
                        continue
                    for word_id in cell_rel["Ids"]:
                        word = block_map.get(word_id, {})
                        if word.get("BlockType") == "WORD" and "Text" in word:
                            words.append(word["Text"])
                        elif word.get("BlockType") == "SELECTION_ELEMENT":
                            #Checkbox → boolean string
                            words.append(
                                "yes" if word.get("SelectionStatus") == "SELECTED"
                                else "no"
                            )

                cells[(r, c)] = " ".join(words).strip()

        #Build dense 2D grid (empty string for any missing cell position)
        grid: list[list[str]] = [
            ["" for _ in range(max_col + 1)]
            for _ in range(max_row + 1)
        ]
        for (r, c), text in cells.items():
            grid[r][c] = text

        tables.append(grid)

    return tables


def _is_label_row(row: list[str]) -> bool:
    """
    This function returns True if the row looks like column label headers
    rather than student data. Used to skip the first row of the student
    table if Textract detects the rotated column label headers as a data row.
    """
    _LABELS = {"name", "names", "lastname", "firstname", "age", "address", "sex", "dob"}
    return any(cell.strip().lower() in _LABELS for cell in row)
