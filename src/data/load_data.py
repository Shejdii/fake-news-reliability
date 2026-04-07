# Load data for fake news reliability project from Hugging Face datasets library
from datasets import load_dataset


def load_fake_news_data():
    dataset = load_dataset("ucsbnlp/liar", trust_remote_code=True)

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data
