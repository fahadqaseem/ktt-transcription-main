import os
import typer
import pandas as pd
from .engine import DeepLinkageEngine
from .trainer import DeepLinkageTrainer
from .config import DeepERConfig

app = typer.Typer(help="Deep Learning Entity Resolution CLI")

# Instantiate config for default fallback values
_config = DeepERConfig()


@app.command()
def train(
        dataset_path: str = typer.Option(
            _config.default_data_path,
            "--data",
            "-d",
            help="Path to raw OCR CSV for training",
        )
):
    """Trains a custom Deep Learning model on your specific OCR data."""
    df = pd.read_csv(dataset_path)
    config = DeepERConfig()
    trainer = DeepLinkageTrainer(config)
    trainer.train_on_data(df)


@app.command()
def link(
        dataset_path: str = typer.Option(
            _config.default_data_path,
            "--data",
            "-d",
            help="Path to OCR CSV to link",
        ),
        out_dir: str = typer.Option(
            "outputs/",
            "--out",
            "-o",
            help="Output directory"
        ),
        use_custom_model: bool = typer.Option(
            True,
            "--custom/--base",
            help="Use custom trained model vs base sentence-transformer"
        ),
):
    """Links records together using FAISS and Vector Embeddings."""
    df = pd.read_csv(dataset_path)
    config = DeepERConfig(distance_threshold=0.15)

    if use_custom_model:
        config.model_name = config.custom_model_out_path

    os.makedirs(out_dir, exist_ok=True)

    engine = DeepLinkageEngine(config)
    linked_df = engine.run_pipeline(df)

    out_file = os.path.join(out_dir, "deep_linked_records.csv")
    linked_df.to_csv(out_file, index=False)
    typer.secho(f"Success! Linked data saved to {out_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()