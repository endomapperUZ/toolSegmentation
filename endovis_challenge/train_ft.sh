#!/bin/bash

cd /home/clara/Documentos/TFM/toolSegmentation/endovis_challenge
pip install albumentations

python3 train_ft_exec.py \
    --model_path data/models/linknet_binary_20/model_0.pt \
    --model_type LinkNet34 \
    --model_name linknet \
    --num_mod 100 \
    --type_mod ent \
    --jaccard-weight 0.3 \
    --fold 0 \
    --root /home/clara/Documentos/TFM/toolSegmentation/endovis_challenge \
    --batch-size 2 \
    --n-epochs 30 \
    --lr 0.01 \
    --workers 1 \
    --num_classes 1 \
    --train_size 0.8 

python3 train_ft_exec.py \
    --model_path data/models/linknet_binary_20/model_0.pt \
    --model_type LinkNet34 \
    --model_name linknet \
    --num_mod 100 \
    --type_mod ent \
    --jaccard-weight 0.3 \
    --fold 0 \
    --root /home/clara/Documentos/TFM/toolSegmentation/endovis_challenge \
    --batch-size 2 \
    --n-epochs 40 \
    --lr 0.001 \
    --workers 1 \
    --num_classes 1 \
    --train_size 0.8 

