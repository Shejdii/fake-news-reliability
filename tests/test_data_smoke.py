from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def test_preprocessed_data_contract(monkeypatch, sample_liar_dataset):
    def mock_load_dataset(*args, **kwargs):
        return sample_liar_dataset

    monkeypatch.setattr("src.data.load_data.load_dataset", mock_load_dataset)

    train_data, _, _ = load_fake_news_data()
    processed = preprocess_dataset(train_data)

    assert processed.column_names == ["text", "label"]

    sample = processed[0]
    assert isinstance(sample["text"], str)
    assert isinstance(sample["label"], int)
