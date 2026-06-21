"""Canonical field names for predictions and ground truth."""

# Keep your original lists exactly as they are to preserve downstream compatibility
HEADER_COLUMNS = [
    "school",
    "grade",
    "teacher",
    "Date_of_Record",
    "schoolyear",
    "semester",
]

ROW_COLUMNS = [
    "lastname",
    "firstname",
    "Sex",
    "age",
    "address",
    "town",
    "comment",
    "days_absent",
    "vaccination",
    "note",
    "Date_Left",
    "DoB",
]

ALL_COLUMNS = [*HEADER_COLUMNS, *ROW_COLUMNS]

DEFAULT_ROW_MATCH_COLUMNS = [
    "lastname",
    "firstname",
    "age",
    "address",
    "town",
    "DoB",
]

# =====================================================================
# PERFORMANCE UPGRADES: Immutable O(1) Sets for Hot Loop Evaluations
# =====================================================================
# Pre-calculating these saves the engine from constantly casting lists to sets
# inside multi-core background worker tasks.

HEADER_COLUMNS_SET = frozenset(HEADER_COLUMNS)
ROW_COLUMNS_SET = frozenset(ROW_COLUMNS)
ALL_COLUMNS_SET = frozenset(ALL_COLUMNS)
DEFAULT_ROW_MATCH_COLUMNS_SET = frozenset(DEFAULT_ROW_MATCH_COLUMNS)