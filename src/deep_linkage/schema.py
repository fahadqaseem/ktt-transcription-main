# schema.py
import pandas as pd


def flatten_record(row: pd.Series) -> str:
    """
    Standardized schema serialization shared by BOTH trainer.py and engine.py.
    Ensures zero formatting drift between contrastive training and FAISS inference.
    """
    # Clean string fields
    firstname = str(row.get("firstname", "")).strip()
    lastname = str(row.get("lastname", "")).strip()
    name = f"{firstname} {lastname}".strip()

    address = str(row.get("address", "")).strip()
    town = str(row.get("town", "")).strip()
    school = str(row.get("school", "")).strip()

    # Clean numeric fields (cast floats like 11.0 or 1904.0 to integer strings '11', '1904')
    age = ""
    if (
        pd.notna(row.get("age"))
        and str(row.get("age")).strip()
        not in ["", "UNKNOWN", "nan"]
    ):
        try:
            age = f"Age:{int(float(row['age']))}"
        except ValueError:
            age = f"Age:{row['age']}"

    year = ""
    if (
        pd.notna(row.get("schoolyear"))
        and str(row.get("schoolyear")).strip()
        not in ["", "UNKNOWN", "nan"]
    ):
        try:
            year = f"Year:{int(float(row['schoolyear']))}"
        except ValueError:
            year = f"Year:{row['schoolyear']}"

    # Build consistent delimiter token string
    parts = [p for p in [name, address, town, age, year, school] if p]
    return " | ".join(parts)
