from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_baseline_features(
    train_data,
    test_data,
    max_features=5000,
    ngram_range=(1, 1),
):
    x_train_text = train_data["text"]
    y_train = train_data["label"]

    x_test_text = test_data["text"]
    y_test = test_data["label"]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )

    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    return x_train, x_test, y_train, y_test, vectorizer