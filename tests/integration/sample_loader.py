import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_sample_liar_dataset() -> DatasetDict:
    """Load a tiny local LIAR-like dataset for deterministic tests."""
    fixture_path = Path(__file__).parent / "liar_sample.json"

    with fixture_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return DatasetDict(
        {split_name: Dataset.from_list(rows) for split_name, rows in data.items()}
    )
