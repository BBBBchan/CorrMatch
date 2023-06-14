#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
checkpoint_path=your/checkpoint/path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    evaluate.py \
    --config=$config --checkpoint_path $checkpoint_path