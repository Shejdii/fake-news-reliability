"""Utilities for balancing class distributions in Hugging Face datasets."""

import random
from collections import defaultdict


def balance_dataset(dataset, label_column="label", target_count=None, seed=42):
    """Downsample dataset classes to a shared target count."""
    label_to_indices = defaultdict(list)

    for idx, label in enumerate(dataset[label_column]):
        label_to_indices[label].append(idx)

    class_counts = {label: len(indices) for label, indices in label_to_indices.items()}

    if target_count is None:
        target_count = min(class_counts.values())

    random.seed(seed)
    balanced_indices = []

    for indices in label_to_indices.values():
        if len(indices) > target_count:
            sampled = random.sample(indices, target_count)
        else:
            sampled = indices

        balanced_indices.extend(sampled)

    balanced_dataset = dataset.select(sorted(balanced_indices)).shuffle(seed=seed)
    return balanced_dataset
