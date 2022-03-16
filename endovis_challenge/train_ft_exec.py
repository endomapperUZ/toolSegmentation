import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi
from generate_masks import get_model
import os
import torch
import torch.optim as optim
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from loss import LossBinary, LossMulti
from dataset_ft import RigidDataset
import utils_ft
import sys
from prepare_train_val_ft import get_split
from types import SimpleNamespace

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

folds = ["train_data2/train_raw"]
train_heights = [1056]
train_widths = [1280]
val_heights = [1056]
val_widths = [1280]

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path',type=str)
    arg('--model_type',type=str)
    arg('--model_name',type=str)
    arg('--num_mod',type=int)
    arg('--type_mod',type=str)
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--fold', type=int)
    arg('--root', type=str)
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--num_classes', type=int)
    arg('--train_size', type=float)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0" 
    model = get_model(args.model_path, model_type=args.model_type, problem_type='binary')
    loss = LossBinary(args.jaccard_weight)

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RigidDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    fold = folds[args.fold]
    train_crop_height = train_heights[args.fold]
    train_crop_width = train_widths[args.fold]
    val_crop_height = val_heights[args.fold]
    val_crop_width = val_widths[args.fold]

    train_file_names, val_file_names = get_split(args.root, fold, args.train_size)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            #PadIfNeeded(min_height=train_crop_height, min_width=train_crop_width, p=1),
            RandomCrop(height=train_crop_height, width=train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            #PadIfNeeded(min_height=val_crop_height, min_width=val_crop_width, p=1),
            CenterCrop(height=val_crop_height, width=val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type='binary',
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type='binary',
                               batch_size=1)
    dataloaders = {"train" : train_loader,"val" : valid_loader}

    valid = validation_binary
    
    utils_ft.train(
        init_optimizer=lambda lr: Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
        args=args,
        model=model,
        model_name = args.model_name , 
        num_mod = args.num_mod,
        type_mod = args.type_mod,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=fold,
        num_classes=args.num_classes
    )

if __name__ == '__main__':
    main()
