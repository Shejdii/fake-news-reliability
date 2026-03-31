from sklearn.metrics import accuracy_score, classification_report, f1_score
import mlflow
import mlflow.sklearn

from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset
from src.features.baseline_features import prepare_baseline_features
from src.models.baseline_model import train_baseline_model

import warnings
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle"
)

def run_baseline_training():
    max_features = 10000
    ngram_range=(1,2)

    # Setting up MLFlow experiment. If the experiment with the given name doesn't exist, it will be created.
    mlflow.set_experiment("fake-news-reliability")

    # Starting a new MLFlow run. The run name is optional, but it can help to identify the run in the MLFlow UI.
    with mlflow.start_run(run_name="baseline_tfidf_logreg_balanced"):
        train_data, test_data, _ = load_fake_news_data()

        train_data = preprocess_dataset(train_data)
        test_data = preprocess_dataset(test_data)

        x_train, x_test, y_train, y_test, vectorizer = prepare_baseline_features(
            train_data=train_data,
            test_data=test_data,
            max_features=max_features,
        )

        model = train_baseline_model(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(y_test, y_pred, digits=4, zero_division=0)

        # Parameters of the model and features.
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("feature_type", "tfidf")
        mlflow.log_param("class_weight", "balanced")

        # Metrics of the model.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Loging model to MLFlow.
        mlflow.sklearn.log_model(model, name="model")

        print("\n=== BASELINE RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"F1 weighted: {f1_weighted:.4f}")

        print("\n=== CLASSIFICATION REPORT ===")
        print(report)

        print("\n=== FEATURE INFO ===")
        print(f"Train shape: {x_train.shape}")
        print(f"Test shape: {x_test.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    run_baseline_training()