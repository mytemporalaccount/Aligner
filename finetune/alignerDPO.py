"""
Value Alignment tuning using Aligner method with Direct Preference Optimization https://arxiv.org/abs/2305.18290
"""

import os
import sys
import time
from pathlib import Path
import shutil

import lightning as L
import numpy as np
import torch

import torch.nn.functional as F

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.aligner import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

devices = 1
using_reference = True

# Hyperparameters
learning_rate = 1e-4
batch_size = 64 / devices
micro_batch_size = 2
num_chosen_rejected_pair = micro_batch_size // 2
beta = 0.1

gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 160000  # train dataset size ; alpaca is 50000, isotonic is 280000, baize is 200000
num_epochs = 3
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
max_seq_length = 1024 #alpaca 256, dolly 1024, lima 2048, isotonic 1536 # see scripts/prepare_alpaca.py
warmup_iters = 0.5 * (epoch_size // micro_batch_size) // devices  # 2 alpaca epochs
start_iter = 0 * (epoch_size // micro_batch_size) // devices  # 2 alpaca epochs

instruction_tuning = True
eval_interval = 0.1 * epoch_size / batch_size
save_interval = 0.5 * epoch_size / batch_size
eval_iters = 200
log_interval = 10

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_iters,
    "zero_optimization": {"stage": 2},
}

model_size = '7B'
aligner_length = 1
aligner_start_layer = 2
data_dir = Path("data/beaver_safe2")
safer_or_better = "safer"
model_base = "lit-llama-2"
model_version = model_size
pretrained_path = Path(f"checkpoints/{model_base}/{model_version}/lit-llama.pth")

## it is recommended by DPO paper that you do a SFT over the same value alignment dataset first and use it as the reference model
## this path is a pretrained aligner model over beaver safe answer that can be used as a reference model
previous_aligner_path = "aligner_weights/PKU-Beaver-DPO-Value-Alignment/BeaverDatasetSafetyPartSFTOnly/aligner1token-llama2-7B-epoch-2.0.pth"

aligner_base_version ="beaver_safe_alpacaStyle_SFT" #"hhSFT" # 
previous_optimizer_path = ""
out_dir = Path(f"out/DPO/aligner/{model_base}-{data_dir.name}/{model_version}/base_{aligner_base_version}-{aligner_length}vector-start_layer{aligner_start_layer}-beta{beta}lr{learning_rate}bs{int(batch_size)}/")
save_model_name = f"{model_base}-{model_version}-{aligner_length}vector-start_layer{aligner_start_layer}-lr{learning_rate}bs{int(batch_size)}.pth"

print(f"Training LLaMA-Aligner with base model {model_base} using {model_version} parameters on the {data_dir.name} dataset, saving to {out_dir}")

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

device_ref = "cuda:0"
device_model = "cuda:1"

def main():

    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy=(DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto"), 
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name(model_size)
    config.block_size = max_seq_length
    config.adapter_prompt_length = aligner_length
    config.adapter_start_layer = aligner_start_layer

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    with fabric.init_module():
        model = LLaMA(config)
        # model.to(device_ref)
        # strict=False because missing keys due to adapter weights not containted in state dict
        model.load_state_dict(checkpoint, strict=False)
        if previous_aligner_path:
            model.load_state_dict(torch.load(previous_aligner_path), strict=False)
    mark_only_adapter_as_trainable(model)

    model.to(device_model)

    print("aligner length: ",model.config.adapter_prompt_length)

    if using_reference:
        with fabric.init_module():
            reference_model = LLaMA(config)
            # strict=False because missing keys due to adapter weights not containted in state dict
            reference_model.load_state_dict(checkpoint, strict=False)
            if previous_aligner_path:
                reference_model.load_state_dict(torch.load(previous_aligner_path), strict=False)
            
            for _, param in reference_model.named_parameters():
                param.requires_grad = False
    reference_model.to(device_ref)


    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if previous_optimizer_path:
        optimizer.load_state_dict(torch.load(previous_optimizer_path))

    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)
    train(fabric, model, optimizer, train_data, val_data, out_dir, reference_model=reference_model)



def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
    reference_model = None,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(start_iter,max_iters):
        epoch = iter_num * micro_batch_size * devices / epoch_size
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        input_ids = input_ids.to(device_model)
        targets = targets.to(device_model)
        input_ids_for_ref = input_ids.clone().to(device_ref)
        reference_logits = reference_model(input_ids_for_ref).to(device_model)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss, other_metrics = loss_fn(logits, targets, reference_logits) 
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss, val_other_metrics = validate(fabric, model, reference_model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                with open(os.path.join(out_dir, "val log.txt"), "a") as file:
                    file.write(f"iter {iter_num}: val loss {val_loss:.6f}\n")
                with open(os.path.join(out_dir, "val metrics.txt"), "a") as file:
                    file.write(f"iter {iter_num}: val loss {val_loss:.6f}\n{val_other_metrics}\n")

            if step_count % save_interval == 0:
                print(f"Saving adapter weights to {out_dir}")
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                aligner_path = os.path.join(out_dir, f"epoch-{epoch:.1f}-iter-{iter_num:06d}.pth")
                optimizer_path = os.path.join(out_dir, f"optimizer-epoch-{epoch:.1f}-iter-{iter_num:06d}.pth")
                save_model_checkpoint(fabric, model, optimizer, aligner_path, optimizer_path)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"epoch-{epoch} iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, model_name:{out_dir}")
            with open(os.path.join(out_dir, "train log.txt"), "a") as file:
                file.write(f"epoch-{epoch} iter {iter_num}: loss {loss.item():.4f}\n")
            with open(os.path.join(out_dir, "train metrics.txt"), "a") as file:
                file.write(f"epoch-{epoch} iter {iter_num}: loss {loss.item():.4f} \n{other_metrics}\n")
        # Save the final checkpoint at the end of training
    aligner_path = os.path.join(out_dir, f"epoch-{epoch:.1f}-iter-{iter_num:06d}.pth")
    optimizer_path = os.path.join(out_dir, f"optimizer-epoch-{epoch:.1f}-iter-{iter_num:06d}.pth")
    save_model_checkpoint(fabric, model,  optimizer, aligner_path, optimizer_path)


def generate_response(model, instruction, input=""):
    tokenizer = Tokenizer("checkpoints/lit-llama-2/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, reference_model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        input_ids_for_ref = input_ids.clone().to(device_ref)
        reference_logits = reference_model(input_ids_for_ref).to(device_model)
        loss, other_metrics = loss_fn(logits, targets, reference_logits)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "How to rob a bank?"
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    with open(os.path.join(out_dir, "sample response log.txt"), "a") as file:
        file.write(f"\n###Instruction\n{instruction}\n###Response\n{output}\n\n")

    model.train()
    return val_loss.item(), other_metrics


def get_logps(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    # loss_mask = (targets != -100)
    # # dummy token; we'll ignore the losses on these tokens later
    # targets[targets == -100] = 0

    logps_all = logits.log_softmax(-1)

    loss_mask = (targets != -1)
    # dummy token; we'll ignore the losses on these tokens later
    targets[targets == -1] = 0

    per_token_logps = logps_all.gather(dim=2, index=targets.unsqueeze(2))
    per_token_logps = per_token_logps.squeeze(2)

    per_token_logps = per_token_logps * loss_mask
    per_token_logps = per_token_logps.sum(-1)
    if not using_reference:
        per_token_logps = per_token_logps / loss_mask.sum(-1)

    chosen_logps = per_token_logps[:num_chosen_rejected_pair]
    rejected_logps = per_token_logps[num_chosen_rejected_pair:]
    return chosen_logps, rejected_logps


def loss_fn(logits, targets, reference_logits=None):
    policy_chosen_logps, policy_rejected_logps = get_logps(logits, targets)
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    if reference_logits is not None:
        reference_chosen_logps, reference_rejected_logps = get_logps(reference_logits, targets)
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = policy_logratios - ref_logratios
    else:
        logits = policy_logratios

    losses = -F.logsigmoid(beta * logits)    


    # get other metrics
    metrics = {}
    if reference_logits is not None:
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'rewards/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
    
    return losses.mean(), metrics
    


def get_batch(fabric: L.Fabric, data: list):
     
    ix = torch.randint(len(data), (num_chosen_rejected_pair,))

    if data_dir.name == "hh":
        input_ids_chosen = [data[i]["chosen"].type(torch.int64) for i in ix]
        labels_chosen = [i.clone() for i in input_ids_chosen]
        # labels_chosen = [data[i]["label_chosen"].type(torch.int64) for i in ix]
        input_ids_rejected = [data[i]["rejected"].type(torch.int64) for i in ix]
        labels_rejected = [i.clone() for i in input_ids_rejected]
        # labels_rejected = [data[i]["label_rejected"].type(torch.int64) for i in ix]
    elif "beaver" in data_dir.name:
        if safer_or_better == 'safer':
            using_index = "safer_response_id"
        elif safer_or_better == "better":
            using_index = "better_response_id"
        input_ids_chosen = [data[i][f"dialog_{data[i][using_index]}"].type(torch.int64) for i in ix]
        labels_chosen = [i.clone() for i in input_ids_chosen]
        input_ids_rejected = [ data[i][f"dialog_{1-data[i][using_index]}"].type(torch.int64) for i in ix]
        labels_rejected = [i.clone() for i in input_ids_rejected]

    max_len = max(max(len(s) for s in input_ids_chosen), max(len(s) for s in input_ids_rejected))

    def pad_right(x, pad_id):
        if len(x) > max_len:
            x = torch.tensor(x, dtype=x.dtype)
            #truncate based on max length
            return x[:max_len]
        else:
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
        
    x_chosen = torch.stack([pad_right(x, pad_id=0) for x in input_ids_chosen])
    x_rejected = torch.stack([pad_right(x, pad_id=0) for x in input_ids_rejected])
    x = torch.cat([x_chosen, x_rejected])

    y_chosen = torch.stack([pad_right(x, pad_id=-1) for x in labels_chosen])
    y_rejected = torch.stack([pad_right(x, pad_id=-1) for x in labels_rejected])
    y = torch.cat([y_chosen,y_rejected])

    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def save_model_checkpoint(fabric, model, optimizer, aligner_path, optimizer_path):
    aligner_path = Path(aligner_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = aligner_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            # and only keep the adapter weights
            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict)
            torch.save(state_dict, aligner_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict())
        if fabric.global_rank == 0:
            torch.save(state_dict, aligner_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        fabric.barrier()


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
