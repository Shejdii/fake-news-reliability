import numpy as np
import torch

from src.confidence.utils import build_prediction_output


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def predict_with_confidence_distilbert(model, tokenizer, text, max_length=256):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits.cpu().numpy()
    probabilities = softmax(logits)[0]

    predicted_label_id = int(np.argmax(probabilities))
    confidence_score = float(np.max(probabilities))

    return build_prediction_output(predicted_label_id, confidence_score)
