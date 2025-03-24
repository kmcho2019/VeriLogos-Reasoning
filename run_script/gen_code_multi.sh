#!/bin/bash

for j in {0..19} # Code indexes
do
    for i in {0..7} # GPUs
    do
        accelerate launch --gpu_ids $i main.py GEN_HDL -im deepseek-ai/deepseek-coder-6.7b-instruct -d evaluation.jsonl -mp True -np 8 -ip $i -i $j &
    done
    wait
done
