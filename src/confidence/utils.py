LABEL_NAMES = {
    0: "FALSE",
    1: "MIXED",
    2: "TRUE",
}


def get_confidence_band(score):
    if score < 0.50:
        return "not_sure"
    if score < 0.80:
        return "fairly_sure"
    return "very_sure"


def build_prediction_output(predicted_label_id, confidence_score):
    predicted_label = LABEL_NAMES[int(predicted_label_id)]

    return {
        "predicted_label": predicted_label,
        "confidence_score": round(float(confidence_score), 4),
        "confidence_band": get_confidence_band(float(confidence_score)),
    }
