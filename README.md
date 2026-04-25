# Fake News Reliability (ML + Confidence Layer)

## What this project does

Classifies statements as:

* FALSE
* MIXED
* TRUE

Built on the LIAR dataset (Hugging Face).

---

## Models

* **Baseline**: TF-IDF + Logistic Regression
* **Transformer**: DistilBERT (fine-tuned)

Tracked with MLflow.

---

## Key Feature — Confidence Layer

Each prediction returns:

```python
{
  "label": "TRUE",
  "confidence": 0.52,
  "band": "fairly_sure"
}
```

Bands:

* `not_sure`
* `fairly_sure`
* `very_sure`

---

## Results (validation sample)

* Baseline accuracy: ~0.42
* DistilBERT accuracy: ~0.59

Insight:

> Higher accuracy ≠ better confidence (DistilBERT not well calibrated)

---

## Run

```bash
make demo-confidence
make compare-confidence
make test
```

---

## Stack

Python, sklearn, transformers, MLflow, pytest, GitHub Actions

---

## Why it matters

Shows:

* model comparison (classic vs transformer)
* experiment tracking (MLflow)
* inference layer design (confidence-aware predictions)
* real-world issue: confidence calibration

## Testing Strategy

- Unit tests run offline using injected dataset loader
- Integration tests (real HF dataset) are marked separately
- CI runs only non-integration tests for stability