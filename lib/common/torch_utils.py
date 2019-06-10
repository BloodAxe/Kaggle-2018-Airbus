import torch
import numpy as np


def logit(x, eps=1e-5):
    x = torch.clamp(x.float(), eps, 1 - eps)
    return torch.log(x / (1 - x))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def one_hot(tensor, num_classes):
    one_hot = torch.zeros(tensor.size() + torch.Size([num_classes]))
    dim = len(tensor.size())
    index = tensor.unsqueeze(dim)
    return one_hot.scatter_(dim, index, 1)


def tensor_from_rgb_image(image):
    image = np.moveaxis(image, -1, 0)
    image = torch.from_numpy(image)
    return image


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise ValueError('Unsupported type')
    return x
