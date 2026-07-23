from __future__ import annotations

from typing import Any
from splink import block_on
import splink.comparison_library as cl


def get_blocking_rules() -> list[Any]:
    """
    Defines cascading blocking rules used for pair generation during
    prediction and Expectation-Maximisation (EM) training.
    """
    return [
        # Broader blocks suitable for small and large datasets
        block_on("lastname"),
        block_on("firstname"),
        "substr(l.lastname, 1, 2) = substr(r.lastname, 1, 2)",
        "l.town = r.town AND l.town IS NOT NULL",
    ]


def get_comparisons() -> list[Any]:
    """
    Comparison levels tuned for historical OCR transcriptions using Splink v4 syntax.
    """
    return [
        # Jaro-Winkler comparisons with threshold arrays passed positionally
        cl.JaroWinklerAtThresholds("lastname", [0.9, 0.7]),
        cl.JaroWinklerAtThresholds("firstname", [0.9, 0.7]),
        cl.JaroWinklerAtThresholds("address", [0.85, 0.65]),

        # Exact match with Term Frequency adjustments via .configure()
        cl.ExactMatch("town").configure(term_frequency_adjustments=True),

        # Levenshtein distance threshold arrays passed positionally
        cl.LevenshteinAtThresholds("DoB", [1, 2]),
        cl.LevenshteinAtThresholds("age", [1, 2]),

        cl.ExactMatch("Sex"),
    ]


def get_splink_settings() -> dict[str, Any]:
    """Generates the master Splink v4 configuration dictionary."""
    return {
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": get_blocking_rules(),
        "comparisons": get_comparisons(),
        "retain_matching_columns": True,
        "retain_intermediate_calculation_columns": False,
        "max_iterations": 100,
        "em_convergence": 0.001,
    }