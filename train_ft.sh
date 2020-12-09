#!/bin/bash
n_epochs=20

for i in 0 1 2
do
    num_epochs_1=$(($n_epochs+10+20*$i))
    python train_ft_exec.py \
        --model_path data/models/linknet_binary_20/model_0.pt \
        --model_type LinkNet34 \
        --model_name linknet \
        --num_mod 0 \
        --type_mod ent \
        --jaccard-weight 0.3 \
        --fold $i \
        --root /content/drive/Shareddrives/TFM_Clara/6_robot-surgery-segmentation-master \
        --batch-size 2 \
        --n-epochs $num_epochs_1 \
        --lr 0.0001 \
        --workers 12 \
        --num_classes 1 \
        --train_size 0.6 

    num_epochs_2=$(($n_epochs+20+20*$i))
    python train_ft_exec.py \
        --model_path data/models/linknet_binary_20/model_0.pt \
        --model_type LinkNet34 \
        --model_name linknet \
        --num_mod 0 \
        --type_mod ent \
        --jaccard-weight 0.3 \
        --fold $i \
        --root /content/drive/Shareddrives/TFM_Clara/6_robot-surgery-segmentation-master \
        --batch-size 2 \
        --n-epochs $num_epochs \
        --lr 0.00001 \
        --workers 12 \
        --num_classes 1 \
        --train_size 0.6 
done