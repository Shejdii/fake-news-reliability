import pytest

from src.data.load_data import load_fake_news_data


@pytest.mark.integration
def test_real_liar_download():
    train_data, test_data, validation_data = load_fake_news_data()

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(validation_data) > 0
