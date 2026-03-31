# Preprocess the LIAR dataset for fake news reliability project
from src.preprocessing.label_mapping import map_label


def preprocess_dataset(dataset):
    dataset = dataset.map(map_label)

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["statement", "target"]]
    )

    dataset = dataset.rename_column("statement", "text")
    dataset = dataset.rename_column("target", "label")

    return dataset
