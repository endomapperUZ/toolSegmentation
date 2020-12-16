#!/bin/bash
n_epochs=20

for i in 0 1 2
do
    num_epochs_1=$(($n_epochs+25+50*$i))
    python train_ft_exec.py \
        --model_path data/models/unet16_binary_20/model_0.pt \
        --model_type UNet16 \
        --model_name unet16 \
        --num_mod 0 \
        --type_mod ent \
        --jaccard-weight 0.3 \
        --fold $i \
        --root /workspace/toolSegmentation \
        --batch-size 8 \
        --n-epochs $num_epochs_1 \
        --lr 0.0001 \
        --workers 3 \
        --num_classes 1 \
        --train_size 0.6 

    num_epochs_2=$(($n_epochs+50+50*$i))
    python train_ft_exec.py \
        --model_path data/models/unet16_binary_20/model_0.pt \
        --model_type UNet16 \
        --model_name unet16 \
        --num_mod 0 \
        --type_mod ent \
        --jaccard-weight 0.3 \
        --fold $i \
        --root /workspace/toolSegmentation \
        --batch-size 8 \
        --n-epochs $num_epochs_2 \
        --lr 0.00001 \
        --workers 3 \
        --num_classes 1 \
        --train_size 0.6 
done
