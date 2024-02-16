import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.aligner import LLaMA
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(
    prompt: str = " How does quantum physics work? \
          [NewPrompt] How can we reduce air pollution? \
          [NewPrompt] Write a short story in third person narration about a protagonist who has to make an important career decision. \
          [NewPrompt] Give three tips for staying healthy. \
          [NewPrompt] What if the Internet had been invented during the Renaissance period? \
            ",
    input: str = "",
    aligner_path: Path = Path("aligner_weights/Alpaca_Instruction_Following_Supervised_Finetuning/aligner10Tokens-llama2-7B-lr0.009bs64weightDecay0.02wu2-epoch-8.0-valloss0.9489.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    question_file = None, 
    is_save_results_to_file = False,
    is_beaver_safety_eval = False,
    quantize: Optional[str] = None, #"llm.int8",
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.7,
    aligner_length: int =1,
    instruct_style: str = "alpaca", # or "alpaca"
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        aligner_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert aligner_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(aligner_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name, aligner_length)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)
    print ("aligner_length:", model.config.adapter_prompt_length)


    tokenizer = Tokenizer(tokenizer_path)

    if is_beaver_safety_eval:
        beaver_safety_eval_question_path = Path("data/evaluation/beaver_safety_eval_questions.json")
        question_file = beaver_safety_eval_question_path
        is_save_results_to_file = True
    
    if question_file:
        samples = extract_prompts_from_file(question_file)
    else:
        samples = extract_prompts_from_input(prompt, input)

    
    for sample in samples:
        prompt = generate_prompt(sample, instruct_style)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        
        print(output)

        output = output.split("### Response:")[1].split("### Instruction")[0].strip()
        sample["response"] = output
        sample["model"] = str(aligner_path)

        tokens_generated = y.size(0) - prompt_length
        print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    if is_save_results_to_file:
        #file path is where the aligner path folder is and the name is the question file name + "_results.json"
        file_path = aligner_path.parent / (question_file.stem + "_results.json")

        with open(file_path, "w") as f:
            json.dump(samples, f, indent=4)
        print("Results saved to ", file_path)


def generate_prompt(example, instruct_style):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    if instruct_style == "alpaca":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    elif instruct_style == "orca":
        return "[|User|] "+ example['instruction']+" [|AI Assistant|] "
    elif instruct_style == "beaver":
        return "[User] "+ example['instruction']+" [Assistant] "
    elif instruct_style == 'hh':
        return "Human: " + example['instruction'] + " Assistant: "
    else:
        raise ValueError(f"Unknown instruction style: {instruct_style}")
    
def extract_prompts_from_file(file_name):
    # load json file
    with open(file_name) as f:
        data = json.load(f)
    # extract prompts
    samples = []
    for d in data:
        try:
            sample = {"instruction": d["prompt"], "input": d["input"]}
        except:
            sample = {"instruction": d["prompt"], "input": ""}
        samples.append(sample)
    return samples

def extract_prompts_from_input(prompts, inputs):
    prompts = prompts.split("[NewPrompt]")
    inputs = inputs.split("[NewPrompt]")
    if len(prompts) == len(inputs):
        samples = [{"instruction": p.strip(), "input": i.strip()} for p, i in zip(prompts, inputs)]
    else:
        samples = [{"instruction": p.strip(), "input": ""} for p in prompts]
    return samples
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
