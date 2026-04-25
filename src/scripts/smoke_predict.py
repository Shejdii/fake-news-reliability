from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("artifacts/models/distilbert_model")
LABELS = {
    0: "FALSE",
    1: "UNCERTAIN",
    2: "TRUE",
}


def main() -> int:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.eval()

    text = "The economy is growing faster than expected this year."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction_id = int(outputs.logits.argmax(dim=-1).item())

    payload = {
        "success": True,
        "prediction_id": prediction_id,
        "prediction_label": LABELS.get(prediction_id, "UNKNOWN"),
    }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
