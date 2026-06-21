"""
ktt-ocr - paddle.py

Description: This is the PaddleOCR adapter for the ktt-ocr-eval pipeline.
             It handles structured extraction of handwritten records from
             scanned document images using local OCR inference.

             Unlike the LLM adapters (GPT, Gemini), PaddleOCR returns raw
             text blocks with bounding boxes — not structured JSON. This
             adapter reconstructs the table structure from coordinates.

             This adapter does two main things:
                 1. Mock/Test mode (USE_MOCK = True) --> Reads from a local fixture file
                                                    inside tests/fixtures/paddle/ with no
                                                    OCR engine required.
                                                    This is for testing the code.
                 2. Live mode (USE_MOCK = False) --> Runs PaddleOCR locally on the
                                                    document image and returns a
                                                    validated OCRPrediction.

                                                    Under the hood:
                                                    - Runs PaddleOCR on the image
                                                    - Flattens detected text blocks
                                                    - Groups blocks into visual rows
                                                      using y-coordinate proximity
                                                    - Splits rows into header area
                                                      and student data table
                                                    - Maps blocks to schema fields
                                                      using x-coordinate ranges
                                                    - Returns a fully assembled
                                                      OCRPrediction object

             Important limitations vs LLM adapters:
                 - Schema fields not present in these documents (Sex, town,
                   comment, note) will always be empty string.
                 - Names are written as "Lastname Firstname" in one column
                   and are split on the first space.
                 - Column header rows (rotated 90°) are filtered out
                   heuristically by x-position density.
                 - Coordinate ranges are tunable via column_ranges config —
                   scans from different angles will need adjustment.

             Registry registration (add to registry.py):
                 from ktt_ocr_eval.adapters.paddle import PaddleOCRAdapter
                 ADAPTERS["paddleocr"] = PaddleOCRAdapter

             YAML method config keys (all optional):
                 lang:          OCR language code           (default: "en")
                 y_tolerance:   Row clustering pixel range  (default: 15)
                 header_cutoff: Fraction of image height
                                above which is page header  (default: 0.28)
                 name_x_max:    Right edge of name column
                                as fraction of image width  (default: 0.21)
                 address_x_min: Left edge of address column
                                as fraction of image width  (default: 0.72)
                 min_confidence: Minimum confidence to keep
                                 a detected text block      (default: 0.5)
                 mock_fixture:  Fixture filename inside
                                tests/fixtures/paddle/      (default: "paddle_mock.json")
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
from __future__ import annotations
import json
import pathlib
import re
from pathlib import Path
from typing import Any

#The third-party imports:
from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig, OCRPrediction


#============================== MOCK TOGGLE ==========================================

#Flip this to False when you're ready to run local OCR.
#When True, no OCR engine is needed and paddlepaddle does not need to be installed.
USE_MOCK: bool = True   # <-- flip to False when you're ready to run real OCR


#============================== FIXTURE PATH =========================================

_FIXTURES_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests" / "fixtures"
)


#============================== CUSTOM EXCEPTION =====================================

#This exception is raised when the PaddleOCR adapter cannot produce a usable response.
#The project runner should catch this and log/skip the document rather than
#letting it crash the whole pipeline.
class PaddleOCRAdapterError(Exception):
    pass


#================================== ADAPTER ==========================================

#This class is the OCR adapter backed by PaddleOCR for local text extraction
class PaddleOCRAdapter:

    #====== Setup ======

    def __init__(self, method: MethodConfig) -> None:
        self.method = method

        #Scalar config, using method.config.get() directly
        #(method.config_mapping() is only for nested dict values)
        self._lang:           str   = str(self.method.config.get("lang", "en"))
        self._y_tolerance:    int   = int(self.method.config.get("y_tolerance", 15))
        self._header_cutoff:  float = float(self.method.config.get("header_cutoff", 0.28))
        self._name_x_max:     float = float(self.method.config.get("name_x_max", 0.21))
        self._address_x_min:  float = float(self.method.config.get("address_x_min", 0.72))
        self._min_confidence: float = float(self.method.config.get("min_confidence", 0.5))
        self._mock_fixture:   str   = str(self.method.config.get("mock_fixture", "paddle_mock.json"))

        #Column ranges for fields between name and address.
        #Expressed as [x_min_fraction, x_max_fraction] of image width.
        #These are tuned for Calumet Public Schools documents at typical scan
        #resolution. Adjust via config_mapping("column_ranges") if scans from
        #a different batch land at different positions.
        _default_ranges: dict[str, list[float]] = {
            "age":        [0.21, 0.25],
            "DoB":        [0.25, 0.33],
            "days_absent":[0.37, 0.42],
            "vaccination":[0.64, 0.69],
            "Date_Left":  [0.69, 0.74],
        }
        self._col_ranges: dict[str, list[float]] = (
            self.method.config_mapping("column_ranges") or _default_ranges
        )

        #Only importing and initialising the OCR engine when we actually need it
        if not USE_MOCK:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise PaddleOCRAdapterError(
                    "paddleocr is not installed. "
                    "Run: pip install paddlepaddle paddleocr"
                ) from exc

            self._ocr = PaddleOCR(
                use_angle_cls = True,
                lang          = self._lang,
                show_log      = False,
            )


    #====== Public ======

    def predict(self, document: DocumentInput) -> OCRPrediction:
        """
        This method runs extraction on the document and returns a validated OCRPrediction.

        When USE_MOCK is True: reads from the local fixture, no OCR engine needed.
        When USE_MOCK is False: runs PaddleOCR locally on the document image.
        Both paths feed the same raw result format into the coordinate-based
        parsing pipeline below.
        """
        if USE_MOCK:
            raw_result = self._get_mock_response()
        else:
            raw_result = self._call_ocr(document)

        lines = _flatten(raw_result, self._min_confidence)

        if not lines:
            #PaddleOCR returned no detections at all — valid but empty result.
            #This is not an error: the runner records it as an empty prediction.
            return OCRPrediction.model_validate({
                "document_id": document.document_id,
                "method":      self.method.name,
                "source_path": str(document.source_path),
                "header":      {},
                "rows":        [],
                "raw_text":    "",
                "raw_output": {
                    "provider":          "paddleocr",
                    "lines":             [],
                    "mock":              USE_MOCK,
                    "document_metadata": document.metadata,
                },
                "metadata": {
                    "adapter": "paddleocr",
                    "lang":    self._lang,
                    "note":    "No text blocks detected.",
                },
            })

        image_width  = max(l["x_right"] for l in lines)
        image_height = max(l["y_bottom"] for l in lines)
        raw_text     = "\n".join(l["text"] for l in lines)

        header_lines, data_lines = _split_header_table(
            lines, image_height, self._header_cutoff
        )
        grouped_rows = _group_rows(data_lines, self._y_tolerance)

        header = self._extract_header(header_lines)
        rows   = self._extract_rows(grouped_rows, image_width)

        return OCRPrediction.model_validate({
            "document_id": document.document_id,
            "method":      self.method.name,
            "source_path": str(document.source_path),
            "header":      header,
            "rows":        rows,
            "raw_text":    raw_text,
            "raw_output": {
                "provider":          "paddleocr",
                "lines":             lines,
                "mock":              USE_MOCK,
                "document_metadata": document.metadata,
            },
            "metadata": {
                "adapter":        "paddleocr",
                "lang":           self._lang,
                "lines_detected": len(lines),
                "rows_extracted": len(rows),
                "mock":           USE_MOCK,
                "note": (
                    "Structure inferred from bounding-box coordinates. "
                    "Fields not present in this document type (Sex, town, "
                    "comment, note) will always be empty string."
                ),
            },
        })


    #====== Private ======

    def _call_ocr(self, document: DocumentInput) -> list:
        """
        This method runs PaddleOCR on the image and returns the raw result.

        PaddleOCR output format:
            [                                       # list of pages (usually 1)
              [                                     # page = list of detections
                [                                   # one detection
                  [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], # bounding box (4 corners)
                  ["text content", 0.98]              # (text, confidence)
                ],
                ...
              ]
            ]
        """
        try:
            result = self._ocr.ocr(str(document.source_path), cls=True)
        except Exception as exc:
            raise PaddleOCRAdapterError(
                f"PaddleOCR engine failed on {document.source_path}: {exc}"
            ) from exc

        if result is None:
            raise PaddleOCRAdapterError(
                f"PaddleOCR returned None for {document.source_path}. "
                "The image may be unreadable or in an unsupported format."
            )

        return result


    def _get_mock_response(self) -> list:
        """
        This method reads a saved PaddleOCR result from the fixture file and
        returns it as a list, matching the exact format that _call_ocr() returns.

        Both paths return the same type (list), so the entire coordinate-based
        parsing pipeline in predict() runs identically in mock and live modes.

        The fixture file is controlled by the "mock_fixture" method config key
        (default: "paddle_mock.json"), so different scenarios can be tested from
        YAML without touching this file:
            config:
              mock_fixture: paddle_mock_sparse.json

        Expected fixture shape:
            [
              [
                [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], ["text", 0.98]],
                ...
              ]
            ]
        """
        fixture_path = _FIXTURES_DIR / self._mock_fixture

        if not fixture_path.exists():
            raise FileNotFoundError(
                f"PaddleOCR mock fixture not found at: {fixture_path}\n"
                "Create the file or set USE_MOCK = False to run real OCR."
            )

        try:
            result = json.loads(fixture_path.read_text(encoding = "utf-8"))
        except json.JSONDecodeError as exc:
            raise PaddleOCRAdapterError(
                f"Fixture is not valid JSON: {exc}"
            ) from exc

        if not isinstance(result, list):
            raise PaddleOCRAdapterError(
                f"Fixture has unexpected shape: expected a list, got {type(result).__name__}. "
                "PaddleOCR output is always a list of pages."
            )

        return result


    def _extract_header(self, header_lines: list[dict]) -> dict[str, str]:
        """
        This method extracts page-level fields from the header area text blocks.

        The Calumet Public Schools header contains:
            - School name (topmost large text)
            - "RECORD OF [section] GRADE [teacher] SCHOOL FOR SEMESTER ENDED [date]"
            - "Assigned to [name], Room No. [number]"

        Extraction is regex-based on the joined header text. It is best-effort:
        the mixed pre-printed + handwritten header is harder for coordinate-based
        OCR than the structured table rows. LLM adapters will outperform here.
        """
        if not header_lines:
            return {}

        #Sort header blocks top-to-bottom, left-to-right and join into one string
        sorted_lines = sorted(header_lines, key=lambda l: (l["y_center"], l["x_left"]))
        full_text = " ".join(l["text"] for l in sorted_lines)

        school   = ""
        grade    = ""
        teacher  = ""
        date_str = ""
        year     = ""
        semester = ""

        #School name: the topmost text block is usually "CALUMET PUBLIC SCHOOLS"
        if sorted_lines:
            school = sorted_lines[0]["text"].strip()

        #Grade: first word after "RECORD OF" that isn't a known keyword
        grade_match = re.search(r"RECORD\s+OF\s+(\w+)", full_text, re.IGNORECASE)
        if grade_match:
            grade = grade_match.group(1).strip()

        #Teacher: name before "SCHOOL FOR SEMESTER"
        teacher_match = re.search(
            r"GRADE\s+(.+?)\s+SCHOOL\s+FOR\s+SEMESTER", full_text, re.IGNORECASE
        )
        if teacher_match:
            teacher = teacher_match.group(1).strip()

        #Date: looks for month abbreviations followed by numbers
        date_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}",
            full_text, re.IGNORECASE
        )
        if date_match:
            date_str = date_match.group(0).strip()

        #Year: four-digit number in the range 1880-1940
        year_match = re.search(r"\b(18[89]\d|19[0-3]\d)\b", full_text)
        if year_match:
            year = year_match.group(1)

        #Semester: infer from the date — Jan/Feb = Spring, Aug/Sep = Fall
        if date_match:
            month = date_match.group(1)[:3].lower()
            semester = "Spring" if month in ("jan", "feb") else (
                "Fall" if month in ("aug", "sep") else ""
            )

        return {
            "school":         school,
            "grade":          grade,
            "teacher":        teacher,
            "Date_of_Record": f"{date_str} {year}".strip(),
            "schoolyear":     year,
            "semester":       semester,
        }


    def _extract_rows(
        self,
        grouped_rows: list[list[dict]],
        image_width: float,
    ) -> list[dict[str, str]]:
        """
        This method maps each grouped data row to a student record dict.

        Strategy:
        - Name:    leftmost block(s) where x_left < name_x_max * image_width
        - Address: rightmost block(s) where x_left > address_x_min * image_width
        - Age, DoB, days_absent, vaccination, Date_Left: blocks whose x-center
          falls within the configured column_ranges fractions of image width.

        Rows with no name block are skipped — they are likely detected column
        header labels or footer lines, not student records.

        Fields not present in this document type (Sex, town, comment, note)
        are always returned as empty string.
        """
        rows: list[dict[str, str]] = []

        name_x_max    = self._name_x_max    * image_width
        address_x_min = self._address_x_min * image_width

        for row_blocks in grouped_rows:
            #Sort blocks left-to-right
            row_blocks = sorted(row_blocks, key=lambda b: b["x_left"])

            #Collect name blocks (leftmost column)
            name_blocks = [b for b in row_blocks if b["x_left"] < name_x_max]
            if not name_blocks:
                #No name block — skip, likely a column header or footer row
                continue

            full_name = " ".join(b["text"] for b in name_blocks).strip()
            lastname, firstname = _parse_name(full_name)

            #Collect address blocks (rightmost column)
            address_blocks = [b for b in row_blocks if b["x_left"] >= address_x_min]
            address = " ".join(b["text"] for b in address_blocks).strip()

            #Map middle blocks to schema fields using x-center fraction
            middle_blocks = [
                b for b in row_blocks
                if b["x_left"] >= name_x_max and b["x_left"] < address_x_min
            ]

            field_values: dict[str, str] = {}
            for block in middle_blocks:
                x_frac = ((block["x_left"] + block["x_right"]) / 2) / image_width
                for field, (x_min, x_max) in self._col_ranges.items():
                    if x_min <= x_frac < x_max:
                        field_values[field] = block["text"]
                        break

            rows.append({
                "lastname":    lastname,
                "firstname":   firstname,
                "Sex":         "",
                "age":         field_values.get("age", ""),
                "address":     address,
                "town":        "",
                "comment":     "",
                "days_absent": field_values.get("days_absent", ""),
                "vaccination": field_values.get("vaccination", ""),
                "note":        "",
                "Date_Left":   field_values.get("Date_Left", ""),
                "DoB":         field_values.get("DoB", ""),
            })

        return rows


#========================== MODULE-LEVEL HELPERS =====================================

def _flatten(raw_result: list, min_confidence: float) -> list[dict[str, Any]]:
    """
    This function converts the nested PaddleOCR output into a flat list of
    block dicts, filtering out low-confidence detections.

    Each output dict contains:
        text      - detected text string
        confidence- detection confidence (0.0-1.0)
        x_left    - left edge of bounding box
        x_right   - right edge of bounding box
        y_top     - top edge of bounding box
        y_bottom  - bottom edge of bounding box
        x_center  - horizontal centre
        y_center  - vertical centre

    PaddleOCR bounding box format:
        [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]  (top-left, clockwise)
    """
    lines: list[dict[str, Any]] = []

    for page in raw_result or []:
        if not page:
            continue
        for detection in page:
            try:
                bbox, (text, confidence) = detection
            except (TypeError, ValueError):
                continue

            if float(confidence) < min_confidence:
                continue

            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]

            x_left   = min(x_coords)
            x_right  = max(x_coords)
            y_top    = min(y_coords)
            y_bottom = max(y_coords)

            lines.append({
                "text":       str(text).strip(),
                "confidence": round(float(confidence), 4),
                "x_left":     x_left,
                "x_right":    x_right,
                "y_top":      y_top,
                "y_bottom":   y_bottom,
                "x_center":   (x_left + x_right) / 2,
                "y_center":   (y_top + y_bottom) / 2,
            })

    return lines


def _split_header_table(
    lines:          list[dict],
    image_height:   float,
    header_cutoff:  float,
) -> tuple[list[dict], list[dict]]:
    """
    This function splits detected lines into page header blocks and table data blocks.

    Everything above header_cutoff * image_height is treated as the page
    header area (school name, record-of line, column labels).
    Everything below is treated as student data rows.
    """
    cutoff_y = header_cutoff * image_height
    header_lines = [l for l in lines if l["y_center"] <= cutoff_y]
    data_lines   = [l for l in lines if l["y_center"] >  cutoff_y]
    return header_lines, data_lines


def _group_rows(lines: list[dict], y_tolerance: int) -> list[list[dict]]:
    """
    This function clusters text blocks into visual rows using y-coordinate proximity.

    Blocks whose y-centres are within y_tolerance pixels of the running row
    mean are merged into the same row. This handles slight vertical misalignment
    between columns in the same student record row.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: l["y_center"])
    buckets: list[list[dict]] = [[sorted_lines[0]]]

    for line in sorted_lines[1:]:
        row_mean_y = sum(l["y_center"] for l in buckets[-1]) / len(buckets[-1])
        if abs(line["y_center"] - row_mean_y) <= y_tolerance:
            buckets[-1].append(line)
        else:
            buckets.append([line])

    return buckets


def _parse_name(full_name: str) -> tuple[str, str]:
    """
    This function splits a "Lastname Firstname" string into its two components.

    In the Calumet Public Schools documents, names are written last-name-first
    with no comma separator (e.g. "Bone Dorothy", "Kowalski Anna").
    We split on the first space. If no space is found, the whole string is
    treated as the last name with an empty first name.
    """
    parts = full_name.strip().split(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return full_name.strip(), ""