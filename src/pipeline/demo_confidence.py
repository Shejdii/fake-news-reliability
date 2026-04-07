import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.confidence.baseline_confidence import predict_with_confidence_baseline
from src.confidence.distilbert_confidence import predict_with_confidence_distilbert
from src.confidence.utils import LABEL_NAMES
from src.data.load_data import load_fake_news_data
from src.features.baseline_features import transform_texts_with_vectorizer
from src.preprocessing.preprocess import preprocess_dataset


BASELINE_MODEL_PATH = "artifacts/baseline/model.joblib"
BASELINE_VECTORIZER_PATH = "artifacts/baseline/vectorizer.joblib"

DISTILBERT_MODEL_PATH = "artifacts/distilbert/model"
DISTILBERT_TOKENIZER_PATH = "artifacts/distilbert/tokenizer"

NUM_EXAMPLES = 5


def load_baseline_artifacts():
    model = joblib.load(BASELINE_MODEL_PATH)
    vectorizer = joblib.load(BASELINE_VECTORIZER_PATH)
    return model, vectorizer


def load_distilbert_artifacts():
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_TOKENIZER_PATH)
    return model, tokenizer


def run_demo():
    baseline_model, baseline_vectorizer = load_baseline_artifacts()
    distilbert_model, distilbert_tokenizer = load_distilbert_artifacts()

    _, _, valid_dataset = load_fake_news_data()
    valid_dataset = preprocess_dataset(valid_dataset)
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(NUM_EXAMPLES))

    print("\n=== CONFIDENCE DEMO ON REAL VALIDATION EXAMPLES ===")

    for idx, row in enumerate(valid_dataset, start=1):
        text = row["text"]
        true_label = LABEL_NAMES[row["label"]]

        baseline_features = transform_texts_with_vectorizer(
            texts=[text],
            vectorizer=baseline_vectorizer,
        )
        baseline_result = predict_with_confidence_baseline(
            model=baseline_model,
            features=baseline_features,
        )

        distilbert_result = predict_with_confidence_distilbert(
            model=distilbert_model,
            tokenizer=distilbert_tokenizer,
            text=text,
        )

        print(f"\nExample {idx}")
        print(f"True label: {true_label}")
        print(f"Text: {text}")

        print("\nBaseline:")
        print(baseline_result)

        print("\nDistilBERT:")
        print(distilbert_result)


if __name__ == "__main__":
    run_demo()