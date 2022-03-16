#!/bin/bash

python3 train_ft.py \
    --path /home/clara/Documentos/TFM/mininet \
    --n_classes 2 \
    --batch_size 2 \
    --epochs 20 \
    --init_lr 1e-3 \
    --crop_factor_x 1 \
    --crop_factor_y 1 \
    --zoom_augmentation 0
