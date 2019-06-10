import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer


def save_snapshot(model: torch.nn.Module, optimizer: Optimizer, loss: float, epoch: int, train_history: pd.DataFrame, snapshot_file: str, multi_gpu=False):
    torch.save({
        'model': model.module.state_dict() if multi_gpu else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_history': train_history.to_dict(),
        'args': ' '.join(sys.argv[1:])
    }, snapshot_file)


def restore_snapshot(model: torch.nn.Module, optimizer: Optimizer, snapshot_file: str, multi_gpu=False):
    checkpoint = torch.load(snapshot_file)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']

    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        model.load_state_dict(checkpoint['model'], strict=False)
        print('Loaded model with strict=False mode')

    try:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('Optimizer state not loaded')

    train_history = pd.DataFrame.from_dict(checkpoint['train_history'])

    return start_epoch, train_history, best_loss


