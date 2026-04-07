from collections import Counter

from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def main():
    train_dataset, _, valid_dataset = load_fake_news_data()

    train_dataset = train_dataset.shuffle(seed=42).select(range(4000))
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(1000))

    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    train_counts = Counter(train_dataset["label"])
    valid_counts = Counter(valid_dataset["label"])

    print("Train class distribution:", train_counts)
    print("Valid class distribution:", valid_counts)

    print("\nSample rows:")
    for i in range(5):
        print(f"\nRow {i}")
        print("label:", train_dataset[i]["label"])
        print("text:", train_dataset[i]["text"][:300])


if __name__ == "__main__":
    main()