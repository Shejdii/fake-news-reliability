from src.preprocessing.label_mapping import map_label


def test_map_label_false_for_pants_fire():
    example = {"label": 0}
    result = map_label(example)
    assert result["target"] == 0


def test_map_label_false_for_false():
    example = {"label": 1}
    result = map_label(example)
    assert result["target"] == 0


def test_map_label_false_for_barely_true():
    example = {"label": 2}
    result = map_label(example)
    assert result["target"] == 0


def test_map_label_uncertain_for_half_true():
    example = {"label": 3}
    result = map_label(example)
    assert result["target"] == 1


def test_map_label_true_for_mostly_true():
    example = {"label": 4}
    result = map_label(example)
    assert result["target"] == 2


def test_map_label_true_for_true():
    example = {"label": 5}
    result = map_label(example)
    assert result["target"] == 2
