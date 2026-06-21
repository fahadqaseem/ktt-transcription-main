from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ktt_ocr_eval.core.artifacts import read_predictions
from ktt_ocr_eval.evaluation.report import evaluate_predictions, write_report
from ktt_ocr_eval.orchestration.methods import load_eval_config, load_methods
from ktt_ocr_eval.orchestration.runner import load_manifest, run_matrix

app = typer.Typer(help="Run and evaluate OCR extraction methods.")


@app.command()
def run(
    manifest: Annotated[Path, typer.Option(help="CSV with document_id and source_path columns.")],
    methods: Annotated[Path, typer.Option(help="YAML method matrix.")],
    out: Annotated[Path, typer.Option(help="Run output directory.")],
    only_method: Annotated[
        str | None,
        typer.Option(help="Run one configured method by name."),
    ] = None,
    no_resume: Annotated[
        bool,
        typer.Option(help="Do not skip completed method/document pairs."),
    ] = False,
) -> None:
    documents = load_manifest(manifest)
    method_configs = load_methods(methods)
    predictions_path = run_matrix(
        documents,
        method_configs,
        out,
        only_method=only_method,
        resume=not no_resume,
    )
    typer.echo(f"Wrote predictions to {predictions_path}")


@app.command()
def eval(
    predictions: Annotated[Path, typer.Option(help="Prediction JSONL from ktt-ocr run.")],
    ground_truth: Annotated[Path, typer.Option(help="Ground truth CSV.")],
    out: Annotated[Path, typer.Option(help="Report output directory.")],
    config: Annotated[Path | None, typer.Option(help="Optional evaluation YAML config.")] = None,
) -> None:
    prediction_items = read_predictions(predictions)
    eval_config = load_eval_config(config)
    report = evaluate_predictions(prediction_items, ground_truth, eval_config)
    write_report(report, out)
    typer.echo(f"Wrote report to {out}")


if __name__ == "__main__":
    app()
