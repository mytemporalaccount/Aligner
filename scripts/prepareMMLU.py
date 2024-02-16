"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path
import os
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

import datasets

DATA_FILE_NAME = "all.json"
IGNORE_INDEX = -1

def prepare(
    data_folder: Path = Path("data/MMLU"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    test_split_ratio: float = 0.1,
    max_sys_prompt_length: int = 128,
    max_seq_length: int = 512,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Dolly dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    # destination_path.mkdir(parents=True, exist_ok=True)
    file_path = data_folder / DATA_FILE_NAME

    if not os.path.exists(data_folder / DATA_FILE_NAME):
        ds = datasets.load_dataset("cais/mmlu", 'all')
        ds.to_json(file_path)


    # # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [json.loads(line) for line in data]
        # data = json.load(file)


    item_length = []
    for item in data:
        item_length.append(len(item["query"])+len(item["response"]))

    print(f"Loaded a total of {len(data)} samples")
    print("Question Answer Lengths")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(item_length, 25))
    print("50th percentile: ", np.percentile(item_length, 50))
    print("75th percentile: ", np.percentile(item_length, 75))
    print("85th percentile: ", np.percentile(item_length, 85))
    print("95th percentile: ", np.percentile(item_length, 95))
    print("99th percentile: ", np.percentile(item_length, 99))
    print("max one: ", max(item_length))


    # Partition the dataset into train and test
    test_split_size = min(int(len(data) * test_split_ratio),2000)

    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data,
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length, max_sys_prompt_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, data_folder / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length, max_sys_prompt_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, data_folder / "test.pt")


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int, max_sys_prompt_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    conversation = "[|User|] "+line['query']+" [|AI Assistant|] "+line['response']
    encoded_full_conversation = tokenize(tokenizer, conversation, max_length=max_length, eos=False)
    return {
        "dialog_ids": encoded_full_conversation,
        "labels": encoded_full_conversation,
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
