from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def test_preprocessed_data_contract():
    train_data, _, _ = load_fake_news_data()
    processed = preprocess_dataset(train_data)

    sample = processed[0]

    assert "text" in sample
    assert "label" in sample

    assert isinstance(sample["text"], str)
    assert isinstance(sample["label"], int)
    assert sample["label"] in [0, 1, 2]

    assert set(sample.keys()) == {"text", "label"}
