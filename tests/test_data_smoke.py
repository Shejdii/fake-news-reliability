from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def test_data_loading_smoke(sample_liar_dataset):
    def fake_loader(*args, **kwargs):
        return sample_liar_dataset

    train_data, test_data, validation_data = load_fake_news_data(
        dataset_loader=fake_loader
    )

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(validation_data) > 0


def test_preprocess_smoke(sample_liar_dataset):
    def fake_loader(*args, **kwargs):
        return sample_liar_dataset

    train_data, _, _ = load_fake_news_data(dataset_loader=fake_loader)
    processed = preprocess_dataset(train_data)

    assert len(processed) > 0
    assert "text" in processed.column_names
    assert "label" in processed.column_names
