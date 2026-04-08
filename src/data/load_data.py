from datasets import load_dataset


def load_fake_news_data(dataset_loader=load_dataset):
    dataset = dataset_loader("ucsbnlp/liar", trust_remote_code=True)
    return dataset["train"], dataset["test"], dataset["validation"]
