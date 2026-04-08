import pytest
from datasets import Dataset, DatasetDict


@pytest.fixture
def sample_liar_dataset():
    train = Dataset.from_dict(
        {
            "statement": [
                "The sky is green.",
                "Water boils at 100C.",
                "The Earth is flat.",
            ],
            "label": [0, 5, 1],
        }
    )

    test = Dataset.from_dict(
        {
            "statement": [
                "Paris is in France.",
                "Vaccines cause magnetism.",
            ],
            "label": [5, 0],
        }
    )

    validation = Dataset.from_dict(
        {
            "statement": [
                "This is half true.",
                "The moon is made of cheese.",
            ],
            "label": [2, 0],
        }
    )

    return DatasetDict(
        {
            "train": train,
            "test": test,
            "validation": validation,
        }
    )
