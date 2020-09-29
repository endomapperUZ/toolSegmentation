import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi

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

def train(model, model_name, num_mod,type_mod, jaccard_weight, fold, root, batch_size, n_epochs, lr, workers, num_classes, train_size):

    loss = LossBinary(jaccard_weight)

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RigidDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(root, fold, train_size)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, problem_type='binary',
                               batch_size=batch_size)
    valid_loader = make_loader(val_file_names, problem_type='binary',
                               batch_size=1)
    dataloaders = {"train" : train_loader,"val" : valid_loader}

    valid = validation_binary

    args = SimpleNamespace(lr = lr, n_epochs = n_epochs, root = root, batch_size = batch_size)
    
    utils_ft.train(
        init_optimizer=lambda lr: Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
        args=args,
        model=model,
        model_name = model_name , 
        num_mod = num_mod,
        type_mod = type_mod,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=fold,
        num_classes=num_classes
    )


