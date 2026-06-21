import json
from pathlib import Path
import pandas as pd


def build_perfect_test_suite():
    print("[*] Reading human ground truth sheet...")
    gt_path = Path("src/ktt_ocr_eval/test_human_ground_truth.csv")

    if not gt_path.exists():
        raise FileNotFoundError(f"Please run generate_test_suite.py first to create {gt_path}")

    df = pd.read_csv(gt_path, dtype=str).fillna("")

    print("[*] Generating 100% pristine perfect predictions...")
    perfect_output = Path("src/ktt_ocr_eval/test_perfect_predictions.jsonl")

    # Generate the document id identical to the ground truth rules
    df["document_id"] = (
            df["school"].str.strip().str.replace(" ", "_") + "||" +
            df["grade"].str.strip().str.replace(" ", "_") + "||" +
            df["Date_of_Record"].str.strip().str.replace(" ", "_")
    )

    grouped = df.groupby("document_id", sort=False)

    with perfect_output.open("w", encoding="utf-8") as f:
        for doc_id, group in grouped:
            first_row = group.iloc[0]

            header_payload = {
                "school": str(first_row["school"]),
                "grade": str(first_row["grade"]),
                "teacher": str(first_row["teacher"]),
                "Date_of_Record": str(first_row["Date_of_Record"]),
                "schoolyear": str(first_row["schoolyear"]),
                "semester": str(first_row["semester"])
            }

            record_list = []
            for _, row in group.iterrows():
                # Zero noise or alterations injected here: pristine 1-to-1 copy
                pristine_record = {
                    "lastname": str(row["lastname"]),
                    "firstname": str(row["firstname"]),
                    "Sex": str(row["Sex"]),
                    "age": str(row["age"]),
                    "address": str(row["address"]),
                    "town": str(row["town"]),
                    "comment": str(row["comment"]),
                    "days_absent": str(row["days_absent"]),
                    "vaccination": str(row["vaccination"]),
                    "note": str(row["note"]),
                    "Date_Left": str(row["Date_Left"]),
                    "DoB": str(row["DoB"])
                }
                record_list.append(pristine_record)

            document_payload = {
                "document_id": str(doc_id),
                "method": "Perfect_Ground_Truth_Baseline",
                "source_path": f"scans/{doc_id}.pdf",
                "header": header_payload,
                "rows": record_list,
                "raw_text": "Pristine baseline stream",
                "raw_output": None,
                "metadata": {"baseline_test": True}
            }

            f.write(json.dumps(document_payload) + "\n")

    print(f"[+] Pristine baseline predictions written to: {perfect_output}")


if __name__ == "__main__":
    build_perfect_test_suite()