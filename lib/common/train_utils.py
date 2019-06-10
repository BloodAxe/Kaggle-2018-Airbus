import os
import numpy as np
import torch
from torch.autograd import Variable


def get_random_name():
    from lib.common import namesgenerator as ng
    return ng.get_random_name()


def should_quit(experiment_dir: str):
    # Magic check to stop training by placing a file STOP in experiment dir
    if os.path.exists(os.path.join(experiment_dir, 'STOP')):
        os.remove(os.path.join(experiment_dir, 'STOP'))
        return True


def log_learning_rate(writer, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar('train/lr/%d' % i, param_group['lr'], global_step=epoch)


def log_model_graph(writer, model):
    try:
        dummy_input = Variable(torch.rand(13, 3, 512, 512))
        writer.add_graph(model, (dummy_input,))
    except:
        pass


def is_better(score, best_score, mode):
    if mode == 'max':
        return score > best_score
    if mode == 'min':
        return score < best_score

    raise ValueError(mode)


def bbox2dict(box):
    return {'x': box[0], 'y': box[1], 'width': box[2], 'height': box[3]}


def dict2bbox(box):
    return [box['x'], box['y'], box['width'], box['height']]


def get_optimizer(optimizer_name, model_parameters, learning_rate, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, **kwargs)

    if optimizer_name == 'rms':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, **kwargs)

    if optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate, **kwargs)

    raise ValueError(optimizer_name)


def compute_total_loss(losses: dict, weights: dict = None):
    total_loss = None

    for name, loss in losses.items():

        if weights is not None:
            loss = loss * float(weights[name])

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    return total_loss
