# Labeling data for fake news reliability classification
# This module provides a mapping of labels for the fake news reliability classification task.
# Define label mapping for fake news reliability classification from the LIAR dataset
# dataset from loading data module

# Define label mapping for LIAR dataset

# Label mapping for LIAR dataset: 6 classes -> 3 classes

LABEL_MAP = {
    0: 0,  # pants-fire -> FALSE
    1: 0,  # false -> FALSE
    2: 0,  # barely-true -> FALSE
    3: 1,  # half-true -> UNCERTAIN
    4: 2,  # mostly-true -> TRUE
    5: 2,  # true -> TRUE
}


def map_label(example):
    label = example["label"]

    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")

    example["target"] = LABEL_MAP[label]
    return example