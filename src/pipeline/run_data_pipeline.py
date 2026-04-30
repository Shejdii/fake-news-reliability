from src.data.load_data import load_fake_news_data
from src.preprocessing.preprocess import preprocess_dataset


def run_data_pipeline():
    train_data, test_data, validation_data = load_fake_news_data()

    train_data = preprocess_dataset(train_data)
    test_data = preprocess_dataset(test_data)
    validation_data = preprocess_dataset(validation_data)

    labels = [row["label"] for row in train_data]
    label_distribution = {
        0: labels.count(0),
        1: labels.count(1),
        2: labels.count(2),
    }

    print("[data] LIAR dataset loaded and preprocessed")
    print(
        f"[data] train={len(train_data)}, "
        f"validation={len(validation_data)}, "
        f"test={len(test_data)}"
    )
    print(f"[data] train label distribution={label_distribution}")


if __name__ == "__main__":
    run_data_pipeline()
