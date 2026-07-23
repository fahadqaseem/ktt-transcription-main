from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from ktt_ocr_eval.core.artifacts import read_predictions_from_csv
from ktt_linkage.engine import RecordLinkageEngine

app = typer.Typer(
    help="Execute probabilistic entity resolution and record linkage.",
    add_completion=False,
)


@app.command(no_args_is_help=True)
def link(
        predictions: Annotated[
            Path,
            typer.Option(
                "--predictions", "-p",
                help="Target CSV file output generated from OCR execution or ground truth table."
            )
        ],
        out: Annotated[
            Path,
            typer.Option(
                "--out", "-o",
                help="Output target directory where the linked entity matrices are exported."
            )
        ],
        threshold: Annotated[
            float,
            typer.Option(
                "--threshold", "-t",
                help="Match probability threshold (0.0 to 1.0) for clustering identities."
            )
        ] = 0.85,
        memory: Annotated[
            int,
            typer.Option(
                "--memory", "-m",
                help="DuckDB memory limit restriction in GB to prevent out-of-memory crashes."
            )
        ] = 8,
) -> None:
    """Executes the Splink probabilistic linkage matrix across records."""
    if not predictions.is_file():
        raise typer.BadParameter(f"CSV file missing or unreadable at: {predictions}")

    typer.echo(f"[*] Ingesting processing file: {predictions.name}")

    # Inspect CSV header to determine ingestion route
    df_preview = pd.read_csv(predictions, nrows=2)

    if "document_id" in df_preview.columns and "method" in df_preview.columns:
        typer.echo("[*] Parsing structured OCR predictions format...")
        input_data = read_predictions_from_csv(predictions)
    else:
        typer.echo("[*] Detected direct ground truth/tabular dataset. Reading CSV directly...")
        input_data = pd.read_csv(predictions)

    typer.echo(f"[*] Initializing Linkage Engine (Threshold: {threshold}, Memory: {memory}GB)...")
    engine = RecordLinkageEngine(
        match_threshold=threshold,
        memory_limit_gb=memory
    )

    typer.echo("[*] Aligning entity matrices via probabilistic EM algorithms...")
    linked_df = engine.run_linkage(input_data)

    out.mkdir(parents=True, exist_ok=True)
    output_file = out / "linked_records.csv"
    linked_df.to_csv(output_file, index=False)

    typer.secho(f"[+] Linkage execution complete! Entity log exported to: {output_file}",
                fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()