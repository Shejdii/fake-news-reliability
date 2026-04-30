"""Preprocess the LIAR dataset for fake news reliability classification."""

from src.preprocessing.label_mapping import map_label


def preprocess_dataset(dataset):
    """Map LIAR labels, keep model input columns, and normalize column names."""
    dataset = dataset.map(map_label)

    columns_to_keep = ["statement", "target"]
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in columns_to_keep]
    )

    dataset = dataset.rename_column("statement", "text")
    dataset = dataset.rename_column("target", "label")

    return dataset
