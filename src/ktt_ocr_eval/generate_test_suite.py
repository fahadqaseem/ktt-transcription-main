import random
from pathlib import Path
import json
import pandas as pd


def inject_ocr_noise(text: str) -> str:
    """Deliberately distorts text data strings to simulate noisy handwriting OCR mutations."""
    if not text or pd.isna(text):
        return ""

    chars = list(str(text))
    # 15% probability to execute typical handwriting character visual substitutions
    if random.random() < 0.15 and len(chars) > 0:
        idx = random.randint(0, len(chars) - 1)
        # Structural dictionary containing common confusion character pairs
        replacements = {'e': 'o', 'o': 'e', 'l': '1', 'i': '1', 'm': 'n', 'n': 'm', 'S': '5', 'B': '8'}
        if chars[idx] in replacements:
            chars[idx] = replacements[chars[idx]]
        else:
            chars[idx] = random.choice(['x', 'a', 's'])

    # 5% probability to completely drop an element (underextraction gap)
    if random.random() < 0.05 and len(chars) > 1:
        chars.pop(random.randint(0, len(chars) - 1))

    return "".join(chars)


def build_golden_test_suite(excel_path: Path, subset_size_classes: int = 3) -> None:
    """Ingests the master workbook data, builds unique page tracking codes,

    and generates paired pristine ground truth and noisy JSONL prediction streams.
    """
    print(f"[*] Ingesting master human verification workbook: {excel_path.name}")
    df = pd.read_excel(excel_path, sheet_name="SR_All_MU", dtype=str).fillna("")

    # Generate the deterministic page tracing id identifier string
    df["document_id"] = (
            df["school"].str.strip().str.replace(" ", "_") + "||" +
            df["grade"].str.strip().str.replace(" ", "_") + "||" +
            df["Date_of_Record"].str.strip().str.replace(" ", "_")
    )

    # Isolate a localized slice of pages for our testing portion
    unique_docs = df["document_id"].unique()
    test_docs = unique_docs[:subset_size_classes]

    # Extract only our test target dataset records
    gt_subset = df[df["document_id"].isin(test_docs)].copy()

    # ----------------------------------------------------
    # EXPORT THE CANONICAL GROUND TRUTH SHEET (.CSV)
    # ----------------------------------------------------
    gt_output = Path("src/ktt_ocr_eval/test_human_ground_truth.csv")
    gt_output.parent.mkdir(parents=True, exist_ok=True)
    gt_subset.to_csv(gt_output, index=False, encoding="utf-8")
    print(f"[+] Pristine ground truth data target written to: {gt_output} ({len(gt_subset)} rows)")

    # ----------------------------------------------------
    # COMPILE & GROUP CAGED OCR PREDICTIONS FLOW (.JSONL)
    # ----------------------------------------------------
    print("[*] Simulating a noisy handwriting OCR model engine extraction loop...")
    ocr_output = Path("src/ktt_ocr_eval/test_mock_ocr_predictions.jsonl")

    # Group row arrays together by document page context before serialization
    grouped = gt_subset.groupby("document_id", sort=False)

    with ocr_output.open("w", encoding="utf-8") as f:
        for doc_id, group in grouped:
            first_row = group.iloc[0]

            # Construct the shared page-level header dictionary layout
            header_payload = {
                "school": str(first_row["school"]),
                "grade": str(first_row["grade"]),
                "teacher": str(first_row["teacher"]),
                "Date_of_Record": str(first_row["Date_of_Record"]),
                "schoolyear": str(first_row["schoolyear"]),
                "semester": str(first_row["semester"])
            }

            # Build your row data collection arrays
            record_list = []
            for _, row in group.iterrows():
                # Randomly drop a row (Simulate an OCR row detection failure omission)
                if random.random() < 0.05:
                    continue

                # Distort row cells using our cursive typewriter handwriting noise mutations
                pred_record = {
                    "lastname": inject_ocr_noise(row["lastname"]),
                    "firstname": inject_ocr_noise(row["firstname"]),
                    "Sex": str(row["Sex"]),
                    "age": inject_ocr_noise(row["age"]),
                    "address": inject_ocr_noise(row["address"]),
                    "town": str(row["town"]),
                    "comment": str(row["comment"]),
                    "days_absent": str(row["days_absent"]),
                    "vaccination": str(row["vaccination"]),
                    "note": str(row["note"]),
                    "Date_Left": str(row["Date_Left"]),
                    "DoB": str(row["DoB"])
                }
                record_list.append(pred_record)

                # Randomly append double-extracted row records (Simulate OCR line bounding duplicates)
                if random.random() < 0.03:
                    duplicate_record = pred_record.copy()
                    duplicate_record["lastname"] = duplicate_record["lastname"] + "_dup"
                    record_list.append(duplicate_record)

            # Combine everything into the precise OCRPrediction input JSON frame structure
            document_payload = {
                "document_id": str(doc_id),
                "method": "Mock_Handwriting_Model_v1",
                "source_path": f"mock_scans/{doc_id}.pdf",
                "header": header_payload,
                "rows": record_list,
                "raw_text": f"Mock stream for document {doc_id}",
                "raw_output": None,
                "metadata": {"synthetic_test_run": True}
            }

            # Flush sequentially to file lines natively
            f.write(json.dumps(document_payload) + "\n")

    print(f"[+] Distorted OCR predictions natively formatted as JSONL at: {ocr_output}")
    print("[+] Test Suite preparation successfully completed!")


if __name__ == "__main__":
    # Point directly to your primary master source workbook file
    build_golden_test_suite(Path("SR_Complete_MASTER February 2020.xlsx"))