"""Map original LIAR labels into project-level reliability classes."""

LABEL_MAP = {
    0: 0,  # pants-fire -> FALSE
    1: 0,  # false -> FALSE
    2: 0,  # barely-true -> FALSE
    3: 1,  # half-true -> UNCERTAIN
    4: 2,  # mostly-true -> TRUE
    5: 2,  # true -> TRUE
}


def map_label(example):
    """Convert a single LIAR label from 6 classes into 3 classes."""
    label = example["label"]

    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")

    example["target"] = LABEL_MAP[label]
    return example
