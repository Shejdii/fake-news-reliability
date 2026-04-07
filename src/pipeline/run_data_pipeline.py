from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset

from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def run_data_pipeline():
    train_data, test_data, validation_data = load_fake_news_data()

    print("\n=== RAW SAMPLE ===")
    print(train_data[0])

    train_data = preprocess_dataset(train_data)
    test_data = preprocess_dataset(test_data)
    validation_data = preprocess_dataset(validation_data)

    print("\n=== PROCESSED SAMPLE ===")
    print(train_data[0])

    print("\n=== DATASET INFO ===")
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    print(f"Validation size: {len(validation_data)}")

    print("\n=== LABEL DISTRIBUTION (train) ===")
    labels = [x["label"] for x in train_data]
    print(
        {
            0: labels.count(0),
            1: labels.count(1),
            2: labels.count(2),
        }
    )


if __name__ == "__main__":
    run_data_pipeline()
