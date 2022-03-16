#!/bin/bash
  
python3 train_classif.py \
    --path /home/clara/Documentos/TFM/toolSegmentation/mininet \
    --n_classes 2 \
    --batch_size 2 \
    --epochs 10 \
    --init_lr 1e-2 \
    --crop_factor_x 1 \
    --crop_factor_y 1 \
    --zoom_augmentation 0
