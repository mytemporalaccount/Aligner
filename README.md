# Aligner: One Global Token is Worth Millions of Parameters When Aligning LLMs

This project implements the Aligner method from the paper "Aligner : One Global Token is Worth Millions of Parameters When Aligning LLMs". Aligner is an extremely parameter-efficient method for LLM alignment tasks (including Instruction Following SFT and Value Alignment), utilizing as few as 1 to 10 tokens but still achieves matching performance as that of LoRA and LLaMA-Adapters. 

But for reasoning tasks, such as mathematics, a few tokens are insufficient for alignment. Aligner requires a token count nearly equivalent to that of LLaMA-Adapters to achieve similar performance in mathematical tasks.

These findings suggest that LLMs process "style" or "form" distinctly from "reasoning" or "knowledge," with form exerting a global influence on the model's capabilities. Conceptualizing LLMs as linear models, the Aligner method bears resemblance to eigen decomposition, revealing that "form" assumes a significantly low rank. orders of magnitude lower than what LoRA suggests.

In summary, our project offers a highly efficient tool and provides theoretically insightful observations in the field of large language models. 


*(Our paper, accompanied by this code base, is currently under review. To preserve anonymity, we have not included a direct link to the arXiv paper here. However, if you are not a reviewer and wish to explore further details, you may search for our paper titled "Aligner: One Global Token is Worth Millions of Parameters When Aligning LLMs.")*


<div align="center">
<img src=assets/parameters.png alt="Aligner method" width="512"/>
</div>

As it turns out, such design achieves extremely high parameter efficiency for alignment tasks, which we include Instruction-Following task and Value Alignment tasks as the representive alignment tasks. Even for reasoning tasks, it is not less efficient. 

## Navigating the Codebase
Our code is based on [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) framework, please refer to it for how to use this framework. 

Most of the guides are in the [howto](howto)

# Set Up and Running
## Setup
Clone the repo

```bash
git clone git@github.com:mytemporalaccount/Aligner.git
cd Aligner
```

install dependencies

```bash
pip install -r requirements.txt
```



## Download and Convert the LLaMA-2 Weight
Notice that you need to download the pretrained LLaMA-2 models or similar structure models and convert them to that of lit-llama format to use this code base. Follow [the guide](howto/download_weights.md) for how to.

## Prepare Datasets
The scripts to download and prepare datsets in the [scripts](scripts) folder. e.g the script to download alpaca dataset is [scripts/prepare_alpaca.py](scripts/prepare_alpaca.py)
To prepare the PKU-Beaver safety dataset for value alignment task, the script is [scripts/prepare_beaver_safe.py](scripts/prepare_beaver_safe.py)

## Supervised Finetuning with Aligner
```bash
python finetune/aligner.py
```
You can adjust the hyper parameters, dataset and the checkpoint paths in the python script. 

## Value Alignment using DPO with Aligner
```bash
python finetune/alignerDPO.py
```
Similarly, you can adjust the hyper parameters, dataset and the checkpoint paths in the python script. 

## Infer with Aligner

```bash
python generate/aligner.py
```
We provide some trained Aligner weights in [aligner_weights](aligner_weights) folder that you can directly use. However, you need to obtain the pretrained LLaMA-2 model yourself.

Notice our provided pretrained Aligner checkpoint is trained with LLaMA-2 7B or 13B. It will still work with LLaMA-1 or any same structure model, but it will suffer some performance drop. For other structure or size models, the checkpoint will not work.

# Method
<div align="center">
<img src=assets/method.png alt="Aligner method" width="512"/>
</div>

The Aligner architecture implements a global prefix token paradigm. Within a transformer-based model,
we prepend a shared set of N learnable tokens to which each layer attends. Further details are based on the LLaMA-
Adapter’s design. Attention is computed on these tokens and added back to the original attention, modulated by a
gating factor. In practice, we find that N = 1 often already suffices to generate answers of similar quality level as
that of LoRA or LLaMA-Adapter.


# Model Results
All the following reponses are not cherry-picked but just taken the first few from the benchmark question set.

## Instruction-Following SFT with Alpaca
### Aligner 1 Token Responses
<div align="center">
<img src=assets/responses/aligner1alpaca.png alt="aligner1alpaca" width="512"/>
</div>

### Aligner 10 Token Responses
<div align="center">
<img src=assets/responses/aligner10alpaca.png alt="aligner10alpaca" width="512"/>
</div>

### LoRA Responses (for comparison)
<div align="center">
<img src=assets/responses/loraAlpaca(1).png alt="loraAlpaca1" width="512"/>
<img src=assets/responses/loraAlpaca(2).png alt="loraAlpaca2" width="512"/>
</div>

### LLaMA-Adapter Responses (for comparison)
<div align="center">
<img src=assets/responses/llama-adapter-Alpaca.png alt="llama-adapter-Alpaca" width="512"/>
</div>


### Vicuna Benchmark
**Better or similar performance with hundreds to thousands times less paramters than LoRA**

<div style="text-align: center;">
  <div style="display: inline-flex;">
    <img src="assets/aligner10token-vs-lora-llama2-7B-alpaca-vicuna.png" alt="Aligner10Token-VS-LoRA-Alpaca--llama2-7B" width="256" />
    <img src="assets/aligner10token-vs-llamaAdapter-llama2-7B-alpaca-vicuna.png" alt="Aligner10Token-VS-LoRA-Alpaca--llama2-7B" width="256" />
  </div>
</div>



With as few as 10 tokens (~50k params), Aligner achieves better win-rate than LoRA (~4 million params) and LLaMA-Adapter(~1.2 million params) for Supervised Finetuning Task with Alpaca dataset on Vicuna benchmark evaluated by GPT4. 


<div align="center">
<img src=assets/alpaca-vicuna-complete-comparisons.png alt="alpaca-vicuna-complete-comparisons" width="256"/>
</div>

 More comprehensive comparisons between Aligner and LoRA, LLaMA-Adapter over both 7B and 13B LLaMA-2 models.

## Value Alignment on PKU-Beaver Dataset
### Aligner 1 Token Responses
<div align="center">
<img src=assets/responses/aligner1beaver.png alt="aligner1beaver" width="512"/>
</div>

### LoRA Responses
<div align="center">
<img src=assets/responses/loraBeaver.png alt="loraBeaver" width="512"/>
</div>

### LLaMA-Adapter Responses
<div align="center">
<img src=assets/responses/llamaAdapterBeaver1.png alt="llamaAdapterBeaver1" width="512"/>
<img src=assets/responses/llamaAdapterBeaver2.png alt="llamaAdapterBeaver2" width="512"/>
</div>

### PKU-Beaver Benchmark

**Better or similar performance with hundreds to thousands times less paramters than LoRA**

<div align="center">
<img src=assets/pku-beaver-alignment-benchmark.png alt="alpaca-vicuna-complete-comparisons" width="512"/>
</div>

On PKU Beaver Safety Benchmark by category, assessed by GPT-4, Aligner with only 1 token performs on par with LLaMA-Adapter and LoRA.

The models are trained with PKU-Beaver dataset using [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) , an alternative of normal RLHF.


## Reasoning Tasks
**Still very powerful with only a few tokens, and no less powerful if deploying same amount of parameters**

### MMLU Benchmark
<div align="center">
<img src=assets/MMLUevalWithAlpacaSFT.png alt="alpaca-vicuna-complete-comparisons" width="256"/>
</div>

When directly evaluating Alpaca-tuned only models on MMLU benchmark, a common sense reasoning task for LLMs, Aligner improves the performance to close to that of LoRA and LLaMA-Adapters with only 10 tokens. 

### Finetuning on MetaMath dataset and Evaluating on GSM8K math benchmark

<div align="center">
<img src=assets/metaMathTuning.png alt="alpaca-vicuna-complete-comparisons" width="356"/>
</div>

The performace increase as the parameter size increases. When Aligner has the same amount of parameters as that of LLaMA-Adapter, it performs just as similar. So Aligner is not less powerful for reasoning tasks.


# Acknowledgement
This code is based on [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) framework. 
Our algorithm is based on [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), and our implementation is also based on the implementation of it in Lit-LLaMA.  