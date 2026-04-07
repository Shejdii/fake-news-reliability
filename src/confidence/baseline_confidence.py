import numpy as np

from src.confidence.utils import build_prediction_output


def predict_with_confidence_baseline(model, features):
    probabilities = model.predict_proba(features)[0]

    predicted_label_id = int(np.argmax(probabilities))
    confidence_score = float(np.max(probabilities))

    return build_prediction_output(predicted_label_id, confidence_score)
