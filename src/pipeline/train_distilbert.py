import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data.load_data import load_fake_news_data
from src.models.distilbert_model import get_distilbert_model_and_tokenizer
from src.preprocessing.balance import balance_dataset
from src.preprocessing.preprocess import preprocess_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
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
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
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
                "eval_accuracy": float(metrics["eval_accuracy"]),
                "eval_f1": float(metrics["eval_f1"]),
            }
        )

        trainer.save_model("artifacts/distilbert/model")
        tokenizer.save_pretrained("artifacts/distilbert/tokenizer")


if __name__ == "__main__":
    train()
