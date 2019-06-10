import random
from typing import Optional

import torch
import pandas as pd
import numpy as np
from torch.optim import Optimizer


def save_checkpoint(snapshot_file: str, model: torch.nn.Module, epoch: int, train_history: pd.DataFrame, optimizer=None, multi_gpu=False, **kwargs):
    dto = {
        'model': model.module.state_dict() if multi_gpu else model.state_dict(),
        'epoch': epoch,
        'train_history': train_history.to_dict(),
        'torch_rng': torch.get_rng_state(),
        'torch_rng_cuda': torch.cuda.get_rng_state_all(),
        'python_rng': random.getstate(),
        'optimizer': optimizer
    }
    dto.update(**kwargs)

    torch.save(dto, snapshot_file)


def restore_checkpoint(snapshot_file: str, model: torch.nn.Module, optimizer: Optional[Optimizer] = None):
    checkpoint = torch.load(snapshot_file, map_location='cpu' if not torch.cuda.is_available() else None)
    start_epoch = checkpoint['epoch'] + 1
    metric_score = checkpoint.get('metric_score', None)

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

    try:
        torch_rng = checkpoint['torch_rng']
        torch.set_rng_state(torch_rng)
        print('Set torch rng state')

        torch_rng_cuda = checkpoint['torch_rng_cuda']
        torch.cuda.set_rng_state(torch_rng_cuda)
        print('Set torch rng cuda state')
    except:
        pass

    try:
        python_rng = checkpoint['python_rng']
        random.setstate(python_rng)
        print('Set python rng state')
    except:
        pass

    train_history = pd.DataFrame.from_dict(checkpoint['train_history'])

    return start_epoch, train_history, metric_score
