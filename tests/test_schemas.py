import pytest

from ktt_ocr_eval.core.schemas import MethodConfig, OCRPrediction


def test_prediction_coerces_numeric_fields_to_strings() -> None:
    prediction = OCRPrediction.model_validate(
        {
            "document_id": "page_001",
            "method": "example",
            "header": {"grade": 3, "schoolyear": 1923},
            "rows": [{"age": 9, "days_absent": 2}],
        }
    )

    assert prediction.header.grade == "3"
    assert prediction.header.schoolyear == "1923"
    assert prediction.rows[0].age == "9"
    assert prediction.rows[0].days_absent == "2"


def test_prediction_allows_missing_header_and_rows() -> None:
    prediction = OCRPrediction.model_validate(
        {
            "document_id": "page_001",
            "method": "example",
            "header": None,
            "rows": None,
        }
    )

    assert prediction.header.canonical()["school"] == ""
    assert prediction.rows == []


def test_prediction_accepts_single_row_object() -> None:
    prediction = OCRPrediction.model_validate(
        {
            "document_id": "page_001",
            "method": "example",
            "rows": {"lastname": "Doe"},
        }
    )

    assert len(prediction.rows) == 1
    assert prediction.rows[0].lastname == "Doe"


def test_method_config_defaults_null_config_to_empty_mapping() -> None:
    method = MethodConfig.model_validate(
        {
            "name": "example",
            "adapter": "example_provider",
            "config": None,
        }
    )

    assert method.config == {}
    assert method.config_mapping("default_header") == {}
    assert method.config_sequence("example_rows") == []


def test_method_config_helpers_raise_clear_type_errors() -> None:
    method = MethodConfig.model_validate(
        {
            "name": "example",
            "adapter": "example_provider",
            "config": {"default_header": [], "example_rows": {}},
        }
    )

    with pytest.raises(TypeError, match="default_header"):
        method.config_mapping("default_header")
    with pytest.raises(TypeError, match="example_rows"):
        method.config_sequence("example_rows")
