import numpy as np

from src.confidence.baseline_confidence import predict_with_confidence_baseline


class DummyBaselineModel:
    def predict_proba(self, features):
        return np.array([[0.12, 0.18, 0.70]])


def test_predict_with_confidence_baseline():
    model = DummyBaselineModel()
    dummy_features = np.array([[1, 2, 3]])

    result = predict_with_confidence_baseline(model, dummy_features)

    assert result["predicted_label"] == "TRUE"
    assert result["confidence_score"] == 0.7
    assert result["confidence_band"] == "fairly_sure"
