from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# Direct architectural linkage to your hyper-optimized modules
from ktt_ocr_eval.core.artifacts import read_predictions
from ktt_ocr_eval.evaluation.report import evaluate_predictions, write_report
from ktt_ocr_eval.orchestration.methods import load_eval_config, load_methods
from ktt_ocr_eval.orchestration.runner import load_manifest, run_matrix

app = typer.Typer(
    help="Run and evaluate high-throughput OCR extraction methods.",
    add_completion=False,
)


@app.command(no_args_is_help=True)
def run(
        manifest: Annotated[
            Path,
            typer.Option(
                "--manifest", "-m",
                help="CSV manifest artifact detailing document_id and source_path entries."
            )
        ],
        methods: Annotated[
            Path,
            typer.Option(
                "--methods", "-e",
                help="YAML configurations specifying your method execution matrix routing."
            )
        ],
        out: Annotated[
            Path,
            typer.Option(
                "--out", "-o",
                help="Destination output directory for your prediction logs."
            )
        ],
        only_method: Annotated[
            str | None,
            typer.Option(
                "--only-method",
                help="Target execution gate to isolate and run a single method layout by its key.",
            ),
        ] = None,
        no_resume: Annotated[
            bool,
            typer.Option(
                "--no-resume",
                help="Force reprocessing by disabling the skip engine for existing document records.",
            ),
        ] = False,
) -> None:
    """Executes the extraction matrix across your configured OCR Adapters pipeline."""
    # Strict path-validation gates protect multi-process workers from silent structural crashes
    if not manifest.is_file():
        raise typer.BadParameter(f"The input CSV manifest file could not be resolved at: {manifest}")
    if not methods.is_file():
        raise typer.BadParameter(f"The active methods matrix YAML file could not be found at: {methods}")

    typer.echo(f"[*] Ingesting processing manifest: {manifest.name}")
    documents = load_manifest(manifest)
    method_configs = load_methods(methods)

    # Fire off your optimized concurrency execution track
    predictions_path = run_matrix(
        documents,
        method_configs,
        out,
        only_method=only_method,
        resume=not no_resume,
    )
    typer.secho(f"[+] Matrix execution complete. Predictions artifact stored at: {predictions_path}",
                fg=typer.colors.GREEN)


@app.command(no_args_is_help=True)
def eval(
        predictions: Annotated[
            Path,
            typer.Option(
                "--predictions", "-p",
                help="Target prediction JSONL file output generated from your ktt-ocr execution path."
            )
        ],
        ground_truth: Annotated[
            Path,
            typer.Option(
                "--ground-truth", "-g",
                help="Master ground truth CSV validation table tracking canonical rows."
            )
        ],
        out: Annotated[
            Path,
            typer.Option(
                "--out", "-o",
                help="Output target directory where evaluation matrices summaries are exported."
            )
        ],
        config: Annotated[
            Path | None,
            typer.Option(
                "--config", "-c",
                help="Optional localized metrics configuration YAML overriding default settings."
            )
        ] = None,
) -> None:
    """Evaluates prediction artifacts against ground truth tables using sparse alignments."""
    # Early-exit checks guarantee data-streams never waste memory parsing blank structures
    if not predictions.is_file():
        raise typer.BadParameter(f"Predictions JSONL file missing or unreadable at: {predictions}")
    if not ground_truth.is_file():
        raise typer.BadParameter(f"Ground Truth tracking table CSV missing at: {ground_truth}")
    if config is not None and not config.is_file():
        raise typer.BadParameter(f"Optional metrics configuration YAML path not found at: {config}")

    typer.echo("[*] Streaming prediction JSONL streams into pre-compiled C validators...")
    if predictions.suffix.lower() == ".csv":
        from ktt_ocr_eval.core.artifacts import read_predictions_from_csv
        prediction_items = read_predictions_from_csv(predictions)
    else:
        prediction_items = read_predictions(predictions)
    
    typer.echo("[*] Mapping configuration weights and routing evaluation keys...")
    eval_config = load_eval_config(config)

    # Dispatches execution directly into your vector-accelerated evaluation engine
    typer.echo("[*] Aligning matrix row weights via Scipy sparse Hungarian algorithms...")
    report = evaluate_predictions(prediction_items, ground_truth, eval_config)

    # Flush metrics summaries using heavy 64KB I/O memory buffers
    write_report(report, out)
    typer.secho(f"[+] Evaluation successfully compiled! Reports written to directory target: {out}",
                fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()