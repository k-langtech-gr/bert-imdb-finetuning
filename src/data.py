import re
from datasets import load_dataset, DatasetDict

def load_imdb(test_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Load IMDb and create train/val/test splits.

    Uses stratified splitting on the training set to create a validation set.
    Also removes HTML line breaks.
    """
    dataset = load_dataset("stanfordnlp/imdb")

    split = dataset["train"].train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column="label",
    )

    def clean(example):
        example["text"] = re.sub(r"<br\s*/?>", " ", example["text"])
        return example

    train_ds = split["train"].map(clean)
    val_ds   = split["test"].map(clean)
    test_ds  = dataset["test"].map(clean)

    return DatasetDict(train=train_ds, val=val_ds, test=test_ds)
