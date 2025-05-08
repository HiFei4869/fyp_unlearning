#!/bin/bash

# Array of hyperparameters
npo_mults=(0.5 0.6 0.7)
rt_mult=0.5

# Loop through different combinations
for npo_mult in "${npo_mults[@]}"; do
    echo "Running experiment with npo_mult=${npo_mult}, rt_mult=${rt_mult}"
    
    uv run src/unlearn.py \
        --model Llama-2-7b-chat \
        --epochs 10 \
        --npo_mult ${npo_mult} \
        --rt_mult ${rt_mult} \
        --save_every -1 \
        --save_model
done