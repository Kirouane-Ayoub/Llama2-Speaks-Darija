import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast

import settings


def get_tokenizer(train_data_path: str, tokenizer_name: str):
    splits = {"v1": "data/v1-00000-of-00001.parquet"}
    df = pd.read_parquet("hf://datasets/ayoubkirouane/Algerian-Darija/" + splits["v1"])
    text = df["Text"].to_list()

    with open(train_data_path, "w") as f:
        f.write("\n".join(text))

    # Function to get training data
    def get_training_corpus():
        dataset = load_dataset("text", data_files={"train": train_data_path})
        for i in range(0, len(dataset["train"]), 1000):
            yield dataset["train"][i : i + 1000]["text"]

    # Initialize the base tokenizer
    base_tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)

    # Train the new tokenizer
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        get_training_corpus(), vocab_size=settings.VOCAB_SIZE
    )

    # Save the new tokenizer
    new_tokenizer.save_pretrained("darija_tokenizer")
    # Add padding token if it doesn't exist
    if new_tokenizer.pad_token is None:
        new_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Ensure pad_token_id is valid
    if (
        new_tokenizer.pad_token_id is None
        or new_tokenizer.pad_token_id >= new_tokenizer.vocab_size
    ):
        new_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        new_tokenizer.pad_token_id = (
            new_tokenizer.vocab_size - 1
        )  # Use the last valid token ID as padding
    return new_tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, encoding="utf-8") as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )  # Ensure PyTorch Tensor output

        input_ids = encoding["input_ids"].squeeze()

        # Assuming you want to use the input_ids as labels for language modeling

        labels = input_ids.clone()

        labels[:-1] = input_ids[1:]  # Shift labels
        return input_ids, labels  # Return both input_ids and labels
