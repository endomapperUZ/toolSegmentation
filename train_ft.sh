#!/bin/bash
n_epochs=80

python train_ft_exec.py \
    --model_path data/models/linknet_binary_20/model_0_ent_15.pt \
    --model_type LinkNet34 \
    --model_name linknet \
    --num_mod 0 \
    --type_mod ent \
    --jaccard-weight 0.1 \
    --fold 0 \
    --root /workspace/ctomasin/toolSegmentation \
    --batch-size 8 \
    --n-epochs 100 \
    --lr 0.0001 \
    --workers 3 \
    --num_classes 1 \
    --train_size 0.84426

python train_ft_exec.py \
    --model_path data/models/linknet_binary_20/model_0_ent_15.pt \
    --model_type LinkNet34 \
    --model_name linknet \
    --num_mod 0 \
    --type_mod ent \
    --jaccard-weight 0.1 \
    --fold 0 \
    --root /workspace/ctomasin/toolSegmentation \
    --batch-size 8 \
    --n-epochs 120 \
    --lr 0.00001 \
    --workers 3 \
    --num_classes 1 \
    --train_size 0.84426

