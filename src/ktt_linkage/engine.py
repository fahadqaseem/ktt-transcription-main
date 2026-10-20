from __future__ import annotations

 import logging
import pandas as pd
from typing import Union

# Splink v4 Modern Imports
from splink import Linker, DuckDBAPI
from splink.internals.exceptions import EMTrainingException

from ktt_ocr_eval.core.schemas import OCRPrediction
from ktt_linkage.config import get_splink_settings, get_blocking_rules

logger = logging.getLogger(__name__)


class RecordLinkageEngine:
    """Robust orchestration layer for training and executing probabilistic linkage."""

    def __init__(self, match_threshold: float = 0.85, memory_limit_gb: int = 8) -> None:
        self.match_threshold = match_threshold
        self.memory_limit = memory_limit_gb
        self.settings = get_splink_settings()

        # Initialize DuckDB Backend
        self.db_api = DuckDBAPI()
        self.db_api._execute_sql_against_backend(f"PRAGMA memory_limit='{self.memory_limit}GB'")
        self.db_api._execute_sql_against_backend("PRAGMA threads=4")

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the dataframe has unique IDs and consistent standard string types."""
        df = df.copy()

        # Generate unique_id if not present
        if "unique_id" not in df.columns:
            df["unique_id"] = [f"row_{i}" for i in range(len(df))]

        # Normalize strings, strip spaces, replace empty/nan strings with None for Splink
        for col in df.columns:
            if col != "unique_id":
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace({"nan": None, "": None, "none": None})

        return df

    def _flatten_predictions(self, predictions: list[OCRPrediction]) -> pd.DataFrame:
        """Flattens nested OCR schemas into a structured tabular format safely."""
        records: list[dict[str, str]] = []

        for pred in predictions:
            header = pred.header.canonical()

            for row_idx, row in enumerate(pred.rows):
                flat_record = {
                    "unique_id": f"{pred.document_id}_{row_idx}",
                    "source_document_id": pred.document_id,
                    **header,
                    **row.canonical(),
                }
                records.append(flat_record)

        return pd.DataFrame(records)

    def run_linkage(self, input_data: Union[list[OCRPrediction], pd.DataFrame]) -> pd.DataFrame:
        """
        Executes full pipeline: Data Prep -> EM Training -> Prediction -> Clustering.
        """
        if isinstance(input_data, list):
            logger.info("📦 Flattening %d OCR prediction documents...", len(input_data))
            df = self._flatten_predictions(input_data)
        elif isinstance(input_data, pd.DataFrame):
            logger.info("📦 Ingesting direct DataFrame table with %d rows...", len(input_data))
            df = input_data
        else:
            raise TypeError("input_data must be a list of OCRPrediction objects or a pandas DataFrame")

        df = self._normalize_dataframe(df)

        if df.empty:
            logger.warning("Empty dataframe provided to linker. Returning empty result.")
            return pd.DataFrame()

        # 1. Initialize Splink Linker
        logger.info("🚀 Initializing Splink v4 Linker Engine on DuckDB backend...")
        linker = Linker(df, self.settings, db_api=self.db_api)

        # 2. Train the Model (Unsupervised EM)
        logger.info("🤖 Phase 1: Estimating initial probabilities via Random Sampling...")
        try:
            linker.training.estimate_u_using_random_sampling(max_pairs=2e6)
        except Exception as e:
            logger.warning("Skipping random sampling u estimation: %s", e)

        logger.info("🤖 Phase 2: Estimating parameters via Expectation-Maximisation (EM)...")
        training_blocks = get_blocking_rules()

        for rule in training_blocks:
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(rule)
            except EMTrainingException:
                logger.warning("⚠️ EM training skipped for rule '%s' (insufficient pairs in dataset).", rule)
            except Exception as e:
                logger.warning("⚠️ Unexpected error during EM training for rule '%s': %s", rule, e)

        # 3. Predict & Cluster
        logger.info("🔗 Clustering matches above %d%% probability...", int(self.match_threshold * 100))
        predictions_df = linker.inference.predict(threshold_match_probability=self.match_threshold)

        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions_df,
            threshold_match_probability=self.match_threshold
        )

        result_df = clusters.as_pandas_dataframe()
        result_df = result_df.rename(columns={"cluster_id": "person_id"})

        logger.info("✅ Successfully identified %d unique individuals from %d records.",
                    result_df["person_id"].nunique(), len(result_df))

        return result_df