# KTT Transcription OCR Evaluation

Local-first harness for evaluating OCR and document extraction methods against
ground truth CSV rows.

The canonical prediction artifact is page-shaped:

- `header`: school/page-level fields used to filter ground truth rows.
- `rows`: student/table rows extracted from the page.
- `raw_text` and `raw_output`: optional debugging artifacts from the method.

## Setup

Required:

- `uv` 0.9+ for dependency management and command execution.
- Python 3.11 or newer. `uv` can install/manage this automatically if needed.

Optional, depending on which methods you run:

- Provider-specific API keys, model weights, or native OCR libraries for the
  adapters you add later.

```bash
uv sync
```

Copy `.env.example` to `.env` if a provider you add later needs environment
variables.

## Run A Method Matrix

```bash
uv run ktt-ocr run \
  --manifest data/manifest.example.csv \
  --methods configs/methods.yaml \
  --out runs/example \
  --only-method example-provider
```

## Evaluate Predictions

```bash
uv run ktt-ocr eval \
  --predictions runs/example/predictions.jsonl \
  --ground-truth data/ground_truth.example.csv \
  --out runs/example/report \
  --config configs/eval.yaml
```

The example provider works without API keys, image files, GPU drivers, or local
OCR model installs.

## Adding OCR Providers

Start from [example_provider.py](src/ktt_ocr_eval/adapters/example_provider.py).
A provider needs an adapter class with a `predict(document)` method that returns
an `OCRPrediction`:

```python
class MyOCRAdapter:
    def __init__(self, method: MethodConfig):
        self.method = method

    def predict(self, document: DocumentInput) -> OCRPrediction:
        return OCRPrediction.model_validate(
            {
                "document_id": document.document_id,
                "method": self.method.name,
                "source_path": str(document.source_path),
                "header": {...},
                "rows": [{...}],
                "raw_text": "...",
                "raw_output": {...},
                "metadata": {"adapter": "my_ocr"},
            }
        )
```

Then register it in [registry.py](src/ktt_ocr_eval/adapters/registry.py) and add
a method entry to [configs/methods.yaml](configs/methods.yaml):

```yaml
methods:
  - name: my-ocr-method
    adapter: my_ocr
    config:
      provider_specific_setting: value
```

You can run the built-in example provider with:

```bash
uv run ktt-ocr run \
  --manifest data/manifest.example.csv \
  --methods configs/methods.yaml \
  --out runs/example-provider \
  --only-method example-provider
```

## Adding Evaluator Metrics

Start from [example_metric.py](src/ktt_ocr_eval/evaluation/example_metric.py).
Custom metrics receive a context dictionary with:

- `prediction`: the original `OCRPrediction`.
- `predicted_rows`: expanded prediction rows with header fields duplicated.
- `expected_rows`: ground truth rows selected by the header filter.
- `filter_status`: whether header filtering matched ground truth.

Register the metric in
[custom_metrics.py](src/ktt_ocr_eval/evaluation/custom_metrics.py), then enable
it in [configs/eval.yaml](configs/eval.yaml):

```yaml
custom_metrics:
  - name: completeness
    metric: example_completeness
    config:
      required_row_fields:
        - lastname
        - firstname
        - age
```

Custom metric values are written to `documents.json` and flattened into
`documents.csv` with a `custom_` prefix.

## Prediction Contract

Header fields:

```text
school, grade, teacher, Date_of_Record, schoolyear, semester
```

Row fields:

```text
lastname, firstname, Sex, age, address, town, comment, days_absent,
vaccination, note, Date_Left, DoB
```

Evaluation expands each predicted row with the predicted header and passes those
records, plus the header-filtered ground truth rows, to configured metrics.
If the ground truth CSV also includes `document_id` and/or `source_path`
columns, evaluation uses them to disambiguate documents before applying the
header filter.
