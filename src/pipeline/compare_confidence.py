import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.confidence.baseline_confidence import predict_with_confidence_baseline
from src.confidence.distilbert_confidence import predict_with_confidence_distilbert
from src.confidence.utils import get_confidence_band
from src.data.load_data import load_fake_news_data
from src.features.baseline_features import transform_texts_with_vectorizer
from src.preprocessing.preprocess import preprocess_dataset


BASELINE_MODEL_PATH = "artifacts/baseline/model.joblib"
BASELINE_VECTORIZER_PATH = "artifacts/baseline/vectorizer.joblib"
DISTILBERT_MODEL_PATH = "artifacts/distilbert/model"
DISTILBERT_TOKENIZER_PATH = "artifacts/distilbert/tokenizer"

NUM_EXAMPLES = 400


def print_summary(model_name, y_true, y_pred, bands):
    print(f"\n=== {model_name.upper()} ===")
    print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Overall f1_weighted: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    for band in ["not_sure", "fairly_sure", "very_sure"]:
        indices = [i for i, value in enumerate(bands) if value == band]
        count = len(indices)

        if count == 0:
            print(f"{band}: count=0")
            continue

        y_true_band = [y_true[i] for i in indices]
        y_pred_band = [y_pred[i] for i in indices]

        acc = accuracy_score(y_true_band, y_pred_band)
        f1 = f1_score(y_true_band, y_pred_band, average="weighted")

        print(
            f"{band}: count={count}, "
            f"accuracy={acc:.4f}, "
            f"f1_weighted={f1:.4f}"
        )


def run_baseline(dataset):
    model = joblib.load(BASELINE_MODEL_PATH)
    vectorizer = joblib.load(BASELINE_VECTORIZER_PATH)

    y_true = []
    y_pred = []
    bands = []

    for row in dataset:
        text = row["text"]
        true_label = row["label"]

        features = transform_texts_with_vectorizer([text], vectorizer)
        result = predict_with_confidence_baseline(model, features)

        label_map = {"FALSE": 0, "MIXED": 1, "TRUE": 2}

        y_true.append(true_label)
        y_pred.append(label_map[result["predicted_label"]])
        bands.append(result["confidence_band"])

    print_summary("baseline", y_true, y_pred, bands)


def run_distilbert(dataset):
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_TOKENIZER_PATH)

    y_true = []
    y_pred = []
    bands = []

    for row in dataset:
        text = row["text"]
        true_label = row["label"]

        result = predict_with_confidence_distilbert(model, tokenizer, text)
        label_map = {"FALSE": 0, "MIXED": 1, "TRUE": 2}

        y_true.append(true_label)
        y_pred.append(label_map[result["predicted_label"]])
        bands.append(result["confidence_band"])

    print_summary("distilbert", y_true, y_pred, bands)


def main():
    _, _, valid_dataset = load_fake_news_data()
    valid_dataset = preprocess_dataset(valid_dataset)
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(NUM_EXAMPLES))

    run_baseline(valid_dataset)
    run_distilbert(valid_dataset)


if __name__ == "__main__":
    main()