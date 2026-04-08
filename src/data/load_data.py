from datasets import load_dataset


def load_fake_news_data(dataset_loader=None):
    if dataset_loader is None:
        dataset_loader = load_dataset

    dataset = dataset_loader("ucsbnlp/liar", trust_remote_code=True)

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data
