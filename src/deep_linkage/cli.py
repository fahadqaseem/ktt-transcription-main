import os
import typer
import pandas as pd
from .engine import DeepLinkageEngine
from .trainer import DeepLinkageTrainer
from .config import DeepERConfig

app = typer.Typer(help="Deep Learning Historical Entity Resolution CLI")

# Instantiate default config for fallback CLI option values
_config = DeepERConfig()


def _load_dataset(filepath: str) -> pd.DataFrame:
    """
    Smart loader that supports both CSVs and Excel sheets (e.g., SR_Complete_MASTER.xlsx).
    """
    if not os.path.exists(filepath):
        typer.secho(f"Error: File not found at {filepath}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    ext = os.path.splitext(filepath)[1].lower()
    typer.secho(f"Loading dataset from: {filepath}...", fg=typer.colors.CYAN)

    if ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(filepath)
        # Load the first sheet by default
        df = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        typer.secho(
            f"Error: Unsupported file format '{ext}'. Use .csv, .xls, or .xlsx.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"Loaded {len(df)} records with columns: {list(df.columns)}",
        fg=typer.colors.GREEN,
    )
    return df


@app.command()
def train(
    dataset_path: str = typer.Option(
        "SR_Complete_MASTER February 2020.xlsx",
        "--data",
        "-d",
        help="Path to raw historical CSV/XLSX for contrastive Triplet training",
    )
):
    """
    Fine-tunes a custom Sentence-Transformer model using 4-layer historical noise
    and household-sibling Hard-Negative Triplet mining.
    """
    df = _load_dataset(dataset_path)
    config = DeepERConfig()

    typer.secho(
        "Initializing DeepLinkageTrainer with Triplet Hard-Negative Mining...",
        fg=typer.colors.YELLOW,
    )
    trainer = DeepLinkageTrainer(config)
    trainer.train_on_data(df)


@app.command()
def link(
    dataset_path: str = typer.Option(
        "SR_Complete_MASTER February 2020.xlsx",
        "--data",
        "-d",
        help="Path to historical CSV/XLSX to link",
    ),
    out_dir: str = typer.Option(
        "outputs/",
        "--out",
        "-o",
        help="Output directory for resolved person_id table",
    ),
    use_custom_model: bool = typer.Option(
        True,
        "--custom/--base",
        help="Use fine-tuned historical model vs default HuggingFace base model",
    ),
    similarity_threshold: float = typer.Option(
        0.85,
        "--threshold",
        "-t",
        help="Cosine similarity threshold for FAISS candidate edges",
    ),
):
    """
    Runs FAISS vector search and Weighted Louvain Modularity clustering
    to assign mathematically stable person_id clusters across school years.
    """
    df = _load_dataset(dataset_path)
    config = DeepERConfig(similarity_threshold=similarity_threshold)

    if use_custom_model:
        config.model_name = config.custom_model_out_path

    os.makedirs(out_dir, exist_ok=True)

    typer.secho(
        f"Initializing DeepLinkageEngine (threshold={config.similarity_threshold}, "
        f"louvain_res={config.louvain_resolution})...",
        fg=typer.colors.YELLOW,
    )
    engine = DeepLinkageEngine(config)

    # Step 1: Encode records into 384d vectors using shared schema.py
    embeddings = engine.encode_records(df)

    # Step 2: Build candidate graph via FAISS IndexFlatIP
    engine.build_candidate_graph(df, embeddings)

    # Step 3: Run Weighted Louvain Community Detection with Bridge Pruning
    linked_df = engine.resolve_entities(df)

    out_file = os.path.join(out_dir, "deep_linked_records.csv")
    linked_df.to_csv(out_file, index=False)
    typer.secho(
        f"✅ Success! Resolved dataset saved to {out_file}",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()