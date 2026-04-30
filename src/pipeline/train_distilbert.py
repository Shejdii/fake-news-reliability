import os
import warnings

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import logging as transformers_logging

from src.data.load_data import load_fake_news_data
from src.models.distilbert_model import get_distilbert_model_and_tokenizer
from src.preprocessing.balance import balance_dataset
from src.preprocessing.preprocess import preprocess_dataset

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but no accelerator is found.*",
)

transformers_logging.set_verbosity_error()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def train():
    train_dataset, _, valid_dataset = load_fake_news_data()

    train_dataset = train_dataset.shuffle(seed=42).select(range(3000))
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(750))

    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    train_dataset = balance_dataset(train_dataset, target_count=1000)

    model, tokenizer = get_distilbert_model_and_tokenizer(
        model_name="distilbert-base-uncased",
        num_labels=3,
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(
        [
            col
            for col in train_dataset.column_names
            if col not in ["input_ids", "attention_mask", "label"]
        ]
    )
    valid_dataset = valid_dataset.remove_columns(
        [
            col
            for col in valid_dataset.column_names
            if col not in ["input_ids", "attention_mask", "label"]
        ]
    )

    train_dataset = train_dataset.rename_column("label", "labels")
    valid_dataset = valid_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="artifacts/distilbert",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        disable_tqdm=True,
        logging_strategy="no",
        logging_steps=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    mlflow.set_experiment("fake-news-reliability")

    with mlflow.start_run(run_name="distilbert_v3_1_moderate_balance"):
        mlflow.log_param("model_name", "distilbert-base-uncased")
        mlflow.log_param("train_size", 3000)
        mlflow.log_param("valid_size", 750)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("balance_target_count", 1000)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("weight_decay", 0.01)

        trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_metrics(
            {
                "eval_loss": float(metrics["eval_loss"]),
                "accuracy": float(metrics["eval_accuracy"]),
                "macro_f1": float(metrics["eval_macro_f1"]),
                "weighted_f1": float(metrics["eval_weighted_f1"]),
                "precision_macro": float(metrics["eval_precision_macro"]),
                "recall_macro": float(metrics["eval_recall_macro"]),
            }
        )

        print("\n=== DISTILBERT RESULTS ===")
        print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"F1 macro: {metrics['eval_macro_f1']:.4f}")
        print(f"F1 weighted: {metrics['eval_weighted_f1']:.4f}")
        print(f"Precision macro: {metrics['eval_precision_macro']:.4f}")
        print(f"Recall macro: {metrics['eval_recall_macro']:.4f}")

        print("\n=== TRAINING INFO ===")
        print("Model: distilbert-base-uncased")
        print("Train size: 3000")
        print("Validation size: 750")
        print("Epochs: 1")
        print("Balance target count: 1000")

        model_dir = "artifacts/models/distilbert_model"

        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        print("\n=== SAVED ARTIFACTS ===")
        print(f"Model saved to: {model_dir}")
        print(f"Tokenizer saved to: {model_dir}")


if __name__ == "__main__":
    train()
