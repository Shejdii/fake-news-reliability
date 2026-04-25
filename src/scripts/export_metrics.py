from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mlflow.tracking import MlflowClient

METRICS_OUT = Path("artifacts/reports/metrics.json")
EXPERIMENT_NAME = "fake-news-reliability"


def get_latest_metrics(experiment_name: str) -> dict[str, float]:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    return runs[0].data.metrics


def require_metric(metrics: dict[str, float], name: str) -> float:
    value = metrics.get(name)

    if value is None:
        raise RuntimeError(
            f"'{name}' not found in MLflow run. "
            f"Available metrics: {list(metrics.keys())}"
        )

    return float(value)


def optional_metric(metrics: dict[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if value is not None else None


def main() -> int:
    metrics = get_latest_metrics(EXPERIMENT_NAME)

    macro_f1 = require_metric(metrics, "macro_f1")

    secondary_metrics = {
        "accuracy": optional_metric(metrics, "accuracy"),
        "weighted_f1": optional_metric(metrics, "weighted_f1"),
        "precision_macro": optional_metric(metrics, "precision_macro"),
        "recall_macro": optional_metric(metrics, "recall_macro"),
    }

    secondary_metrics = {
        key: value for key, value in secondary_metrics.items() if value is not None
    }

    payload = {
        "primary_metric": {
            "name": "macro_f1",
            "value": macro_f1,
        },
        "secondary_metrics": secondary_metrics,
        "metadata": {
            "model_type": "distilbert",
            "dataset": "liar",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
