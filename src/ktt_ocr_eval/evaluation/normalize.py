from __future__ import annotations

import re
import unicodedata
from typing import Any

# OPTIMIZATION 1: Pre-compile the regex pattern once at the module level.
# This moves pattern compilation out of the hot loop entirely.
WHITESPACE_RE = re.compile(r"\s+")


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    # Fast-path for already clean, empty, or whitespace-only inputs
    if not value or (isinstance(value, str) and value.isspace()):
        return ""

    # OPTIMIZATION 2: Direct type check fallback is faster than raw unconditional str() casting
    text = value if isinstance(value, str) else str(value)

    # Unicode decomposition/composition normalization
    text = unicodedata.normalize("NFKC", text)

    # Strip and lower
    text = text.strip().lower()

    # OPTIMIZATION 3: Use the pre-compiled regex object's native .sub() method
    return WHITESPACE_RE.sub(" ", text)


def tokens(value: Any) -> list[str]:
    # OPTIMIZATION 4: Avoid calling normalize_value if the input is completely empty upfront
    if not value:
        return []

    normalized = normalize_value(value)
    if not normalized:
        return []

    # Using .split(" ") is fast, but if you want to split on any whitespace,
    # a blank .split() is optimized in pure C. Sticking to your explicit " " split:
    return normalized.split(" ")