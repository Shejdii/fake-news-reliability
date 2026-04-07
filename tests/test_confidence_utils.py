from src.confidence.utils import build_prediction_output, get_confidence_band


def test_get_confidence_band_not_sure():
    assert get_confidence_band(0.49) == "not_sure"


def test_get_confidence_band_fairly_sure_lower_bound():
    assert get_confidence_band(0.50) == "fairly_sure"


def test_get_confidence_band_fairly_sure_middle():
    assert get_confidence_band(0.72) == "fairly_sure"


def test_get_confidence_band_very_sure():
    assert get_confidence_band(0.80) == "very_sure"


def test_build_prediction_output_false():
    result = build_prediction_output(0, 0.84)

    assert result["predicted_label"] == "FALSE"
    assert result["confidence_score"] == 0.84
    assert result["confidence_band"] == "very_sure"


def test_build_prediction_output_mixed():
    result = build_prediction_output(1, 0.63)

    assert result["predicted_label"] == "MIXED"
    assert result["confidence_score"] == 0.63
    assert result["confidence_band"] == "fairly_sure"


def test_build_prediction_output_true():
    result = build_prediction_output(2, 0.22)

    assert result["predicted_label"] == "TRUE"
    assert result["confidence_score"] == 0.22
    assert result["confidence_band"] == "not_sure"
