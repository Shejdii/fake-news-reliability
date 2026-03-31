from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def test_data_loading_smoke():
    train_data, test_data, validation_data = load_fake_news_data()

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(validation_data) > 0

    sample = train_data[0]
    assert "statement" in sample
    assert "label" in sample


def test_preprocess_smoke():
    train_data, _, _ = load_fake_news_data()
    processed = preprocess_dataset(train_data)

    assert len(processed) == len(train_data)

    sample = processed[0]
    assert "text" in sample
    assert "label" in sample
    assert sample["label"] in [0, 1, 2]