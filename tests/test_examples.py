from pathlib import Path

from ktt_ocr_eval.core.artifacts import read_predictions
from ktt_ocr_eval.evaluation.report import evaluate_predictions
from ktt_ocr_eval.orchestration.methods import load_eval_config, load_methods
from ktt_ocr_eval.orchestration.runner import load_manifest, run_matrix


def test_example_provider_runs(tmp_path: Path) -> None:
    methods = [
        method
        for method in load_methods(Path("configs/methods.yaml"))
        if method.name == "example-provider"
    ]

    predictions_path = run_matrix(
        load_manifest(Path("data/manifest.example.csv")),
        methods,
        tmp_path,
    )
    predictions = read_predictions(predictions_path)

    assert len(predictions) == 1
    assert predictions[0].method == "example-provider"
    assert predictions[0].header.school == "Lincoln School"
    assert predictions[0].rows[0].lastname == "Doe"


def test_example_custom_metric_is_reported(tmp_path: Path) -> None:
    methods = [
        method
        for method in load_methods(Path("configs/methods.yaml"))
        if method.name == "example-provider"
    ]
    predictions_path = run_matrix(
        load_manifest(Path("data/manifest.example.csv")),
        methods,
        tmp_path,
    )

    report = evaluate_predictions(
        read_predictions(predictions_path),
        Path("data/ground_truth.example.csv"),
        load_eval_config(Path("configs/eval.yaml")),
    )

    metric = report["documents"][0]["custom_metrics"]["completeness"]
    assert metric["header_completeness"] == 1.0
    assert metric["row_completeness"] == 1.0
    assert metric["non_empty_row_count"] == 1
