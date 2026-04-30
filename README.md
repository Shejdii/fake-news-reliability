# 📰 Fake News Reliability (MLOps Pipeline)

End-to-end ML pipeline for classifying statement credibility, extended with a confidence-aware prediction layer.

This project focuses on **reproducibility, model comparison, and prediction reliability**, not just accuracy.

---

## ⚡ What this project does

This system:

- loads and preprocesses the LIAR dataset from Hugging Face  
- reduces the original 6-class problem into a **3-class setup**:
  - FALSE
  - MIXED
  - TRUE  
- trains multiple models (baseline + transformer)  
- tracks experiments using MLflow  
- evaluates prediction confidence  
- compares models not only by accuracy, but also by **confidence behavior**  
- runs as a reproducible pipeline via CLI (`make`)

---

## 🎯 Core idea

Raw model predictions are not enough.

A production ML system must answer:

> *How much can we trust this prediction?*

This project introduces a **confidence layer** on top of standard classification.

---

## 🧠 Models

### Baseline

- TF-IDF + Logistic Regression  
- fast and interpretable  
- limited semantic understanding  

### Transformer

- DistilBERT (fine-tuned)  
- captures contextual meaning  
- higher accuracy  

All experiments are tracked with MLflow.

---

## 🔍 Confidence Layer

Each prediction returns:

```json
{
  "label": "TRUE",
  "confidence": 0.52,
  "band": "fairly_sure"
}
```

Confidence is mapped into bands:

- `not_sure`
- `fairly_sure`
- `very_sure`

This makes uncertainty explicit instead of hidden.

---

## 📊 Results (validation subset)

| Model      | Accuracy |
|------------|----------|
| Baseline   | ~0.42    |
| DistilBERT | ~0.59    |

### Key insight

Higher accuracy does **not** imply better reliability.

Observed:

- DistilBERT improves accuracy  
- but confidence scores are **not well calibrated**

This reflects a real-world ML issue:

> reliability ≠ accuracy

---

## ⚙️ Pipeline

Main steps:

1. Data loading (`src/data/load_data.py`)
2. Preprocessing + label mapping
3. Optional dataset balancing
4. Model training:
   - baseline
   - DistilBERT
5. Confidence evaluation
6. Model comparison (`compare_confidence.py`)

Run:

```bash
make pipeline
make demo-confidence
make compare-confidence
make test
```

---

## 🧪 Testing Strategy

- unit tests for:
  - confidence logic
  - label mapping
  - data contracts  
- offline tests using injected dataset loader  
- integration tests (real dataset) separated via markers  

CI runs only stable tests.

---

## 🧰 Stack

- Python  
- scikit-learn  
- transformers  
- MLflow  
- pytest  
- GitHub Actions  

---

## ⚠️ Limitations

- no explicit calibration method (e.g. temperature scaling)  
- confidence derived directly from model probabilities  
- reduced dataset used for faster iteration  

These are known gaps and potential improvements.

---

## 🚀 Why this matters

This project demonstrates:

- comparison of classical ML vs transformer models  
- experiment tracking (MLflow)  
- reproducible ML pipelines  
- separation of training and inference logic  
- **confidence-aware prediction design**

Most ML projects optimize for accuracy only.

This system explicitly models **prediction reliability**, which is critical in real-world ML systems.
