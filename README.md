# anonymous-llm-pruning

## Requirements
We confirmed in A100 GPU with CUDA12.6.

    conda create -n rcpu python=3.10.16
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install transformers datasets
    mkdir tmp_files # under this repo

## Usage
The command like below will start pruning and ppl evaluation.

    CUDA_VISIBLE_DEVICES=0 python main.py --method rcpu --unstr True --nsamples 128 --pruning_ratio 0.2
