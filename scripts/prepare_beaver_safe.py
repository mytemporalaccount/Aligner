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

TRAIN_DATA_FILE_NAME = "train.json"
TEST_DATA_FILE_NAME = "test.json"
IGNORE_INDEX = -1
home_dir = Path("/home/ziheng/ssd-drive1/projects/llm/lit-llama/")
def prepare(
    data_folder: Path = Path("data/beaver_safe"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    test_split_ratio: float = 0.1,
    max_sys_prompt_length: int = 128,
    max_seq_length: int = 256,
    seed: int = 42,
    is_for_debug = False, 
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Dolly dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    # destination_path.mkdir(parents=True, exist_ok=True)
    data_folder = home_dir / data_folder
    train_file_path = data_folder / TRAIN_DATA_FILE_NAME
    test_file_path =  data_folder / TEST_DATA_FILE_NAME

    if not os.path.exists(train_file_path) \
        or not os.path.exists(test_file_path):
        ds = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF")
        ds['train'].to_json(train_file_path)
        ds['test'].to_json(test_file_path)


    # # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(home_dir / tokenizer_path)

    with open(train_file_path, "r") as file:
        data_train = file.readlines()
        data_train = [json.loads(line) for line in data_train]
        # data = json.load(file)
    
    with open(test_file_path, "r") as file:
        data_test = file.readlines()
        data_test = [json.loads(line) for line in data_test]

    response_0_length = []
    response_1_length = []
    for item in data_train:
        response_0_length.append(len(item["response_0"]))
        response_1_length.append(len(item["response_1"]))

    print(f"Loaded a total of {len(data_train)} samples")
    print("chosen")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(response_0_length, 25))
    print("50th percentile: ", np.percentile(response_0_length, 50))
    print("75th percentile: ", np.percentile(response_0_length, 75))
    print("85th percentile: ", np.percentile(response_0_length, 85))
    print("95th percentile: ", np.percentile(response_0_length, 95))
    print("99th percentile: ", np.percentile(response_0_length, 99))
    print("max one: ", max(response_0_length))


    print("rejected")
    print("75th percentile: ", np.percentile(response_1_length, 75))
    print("50th percentile: ", np.percentile(response_1_length, 50))
    print("75th percentile: ", np.percentile(response_1_length, 75))
    print("85th percentile: ", np.percentile(response_1_length, 85))
    print("95th percentile: ", np.percentile(response_1_length, 95))
    print("99th percentile: ", np.percentile(response_1_length, 99))
    print("max one: ", max(response_1_length))


    # Partition the dataset into train and test

    train_set, test_set = list(data_train), list(data_test)
    if is_for_debug:
        train_set = train_set[:10]
        test_set = test_set[:10]

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")


    ## create a new dataset with only safe responses
    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(test_set) if exist_safe(line)
    ]
    torch.save(test_set, data_folder / "test.pt")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(train_set) if exist_safe(line)
    ]
    torch.save(train_set, data_folder / "train.pt")



def exist_safe(line):
    if line['is_response_0_safe'] or line['is_response_1_safe']:
        return True
    else:
        return False

def prepare_line(line, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    dialog_0 = "[User] " + line["prompt"] + " [Assistant] " + line['response_0']
    dialog_1 = "[User] " + line["prompt"] + " [Assistant] " + line['response_1']

    dialog_0 = tokenize(tokenizer, dialog_0, max_length=max_length, eos=False)
    dialog_1 = tokenize(tokenizer, dialog_1, max_length=max_length, eos=False)
    is_resp0_safe = line['is_response_0_safe']
    is_resp1_safe = line['is_response_1_safe']
    better_response_id = line['better_response_id']
    safer_response_id = line['safer_response_id']
    return {
        "dialog_0": dialog_0,
        "dialog_1": dialog_1,
        "is_response_0_safe" : is_resp0_safe,
        "is_response_1_safe" : is_resp1_safe,
        "better_response_id" : better_response_id,
        "safer_response_id" : safer_response_id
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
