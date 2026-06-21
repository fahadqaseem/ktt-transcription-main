from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias, cast, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, BeforeValidator

# OPTIMIZATION 1: Import the high-speed O(1) frozen sets from columns.py
from ktt_ocr_eval.core.columns import HEADER_COLUMNS_SET, ROW_COLUMNS_SET


# High-speed inline modifier function for casting primitives
def _coerce_primitive_to_str(v: Any) -> Any:
    if isinstance(v, (bool, int, float)):
        return str(v)
    return v


# OPTIMIZATION 2: Pre-compiled validator type alias.
# This forces Pydantic to handle primitive string normalization natively via internal loops
# instead of spinning up slow dynamic wildcard runtime hooks over every single field.
OcrString = Annotated[str | None, BeforeValidator(_coerce_primitive_to_str)]

ConfigValue: TypeAlias = str | int | float | bool | None | list[Any] | dict[str, Any]
MethodConfigData: TypeAlias = dict[str, ConfigValue]
Metadata: TypeAlias = dict[str, ConfigValue]
RawOutput: TypeAlias = ConfigValue


class DocumentInput(BaseModel):
    """One source page to process."""

    document_id: str
    source_path: Path
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("document_id")
    @classmethod
    def document_id_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document_id must not be empty")
        return value


class MethodConfig(BaseModel):
    """A configured extraction method from YAML."""

    name: str
    adapter: str
    config: MethodConfigData = Field(default_factory=dict)

    @field_validator("config", mode="before")
    @classmethod
    def default_missing_config(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    def config_mapping(self, key: str) -> dict[str, Any]:
        value = self.config.get(key, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"method config key {key!r} must be a mapping")
        return cast("dict[str, Any]", value)

    def config_sequence(self, key: str) -> list[Any]:
        value = self.config.get(key, [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(f"method config key {key!r} must be a list")
        return value


class PredictionRecord(BaseModel):
    """One extracted student/table row parsed using high-speed native coercion rules."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Fields now use the hyper-optimized type schema annotations
    lastname: OcrString = None
    firstname: OcrString = None
    Sex: OcrString = None
    age: OcrString = None
    address: OcrString = None
    town: OcrString = None
    comment: OcrString = None
    days_absent: OcrString = None
    vaccination: OcrString = None
    note: OcrString = None
    Date_Left: OcrString = None
    DoB: OcrString = None

    def canonical(self) -> dict[str, str]:
        """Direct field dictionary extraction bypassing heavy getattr fallback routines."""
        raw_dict = self.__dict__
        return {
            col: str(raw_dict[col]) if (col in raw_dict and raw_dict[col] is not None) else ""
            for col in ROW_COLUMNS_SET
        }


class PredictionHeader(BaseModel):
    """Extracted page-level fields used to filter ground truth."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    school: OcrString = None
    grade: OcrString = None
    teacher: OcrString = None
    Date_of_Record: OcrString = None
    schoolyear: OcrString = None
    semester: OcrString = None

    def canonical(self) -> dict[str, str]:
        raw_dict = self.__dict__
        return {
            col: str(raw_dict[col]) if (col in raw_dict and raw_dict[col] is not None) else ""
            for col in HEADER_COLUMNS_SET
        }


class OCRPrediction(BaseModel):
    """Canonical page-shaped prediction artifact."""

    document_id: str
    method: str
    source_path: str | None = None
    header: PredictionHeader = Field(default_factory=PredictionHeader)
    rows: list[PredictionRecord] = Field(default_factory=list)
    raw_text: str | None = None
    raw_output: RawOutput = None
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_payload_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        # Performance optimization: conditional modifications to avoid full re-allocations
        if "header" not in value or value["header"] is None:
            value["header"] = {}

        rows = value.get("rows")
        if rows is None:
            value["rows"] = []
        elif isinstance(rows, dict):
            value["rows"] = [rows]
        return value

    def expanded_records(self) -> list[dict[str, str]]:
        """Ultra-fast dictionary unpacking running at pure C speed inside Pydantic loops."""
        header_data = self.header.canonical()
        return [{**header_data, **row.canonical()} for row in self.rows]