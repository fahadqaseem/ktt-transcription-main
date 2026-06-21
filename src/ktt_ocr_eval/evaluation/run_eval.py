from pathlib import Path
from ktt_ocr_eval.core.schemas import OCRPrediction  # Import your actual schema
from report import evaluate_predictions, write_report


def main():
    # 1. Define your file paths
    ground_truth_file = Path("data/ground_truth.csv")
    output_directory = Path("evaluation_outputs")

    # 2. Mock or load your actual parsed OCR predictions list
    # In production, you would loop through your OCR engine outputs to populate this
    dummy_prediction = OCRPrediction(
        document_id="doc_001",
        method="MyModernOCREngine",
        source_path="/path/page1.pdf",
        # ... fill in your internal schema arguments
    )
    predictions_list = [dummy_prediction]

    # 3. Configure the execution run
    evaluation_config = {
        "header_filter_fields": ["invoice_num"],  # Fields used to isolate document chunks
        "evaluated_fields": ["line_item_desc", "total_amount"],  # Rows checked for accuracy
        "match_strategy": "jaccard",  # Triggers our optimized matrix matcher
        "match_threshold": 0.5  # Minimum token overlap to count as a match
    }

    print("🚀 Initializing parallel evaluation highway (Version 4)...")

    # 4. Run the evaluation pipeline
    report_results = evaluate_predictions(
        predictions=predictions_list,
        ground_truth_path=ground_truth_file,
        config=evaluation_config
    )

    # 5. Write the generated analytics records to disk
    print(f"📊 Evaluation complete! Saving reports to: {output_directory}/")
    write_report(report_results, output_directory)


if __name__ == "__main__":
    # Multi-processing guard is MANDATORY in Python when spawning process pools
    main()