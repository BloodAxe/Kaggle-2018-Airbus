import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from lib.losses.segmentation_loss import focal_loss_with_logits


def test_focal_loss():
    a = 1
    g = 2
    x = torch.tensor([-10, 10],dtype=torch.float32)
    y = torch.tensor([0, 1],dtype=torch.float32)

    print(focal_loss_with_logits(x,y, alpha=a, gamma=g), binary_cross_entropy_with_logits(x,y))

    x = torch.tensor([-10, 1],dtype=torch.float32)
    y = torch.tensor([0, 1],dtype=torch.float32)

    print(focal_loss_with_logits(x,y, alpha=a, gamma=g), binary_cross_entropy_with_logits(x,y))

    x = torch.tensor([-10, -10],dtype=torch.float32)
    y = torch.tensor([0, 1],dtype=torch.float32)

    print(focal_loss_with_logits(x,y, alpha=a, gamma=g), binary_cross_entropy_with_logits(x,y))

    x = torch.tensor([10, -10],dtype=torch.float32)
    y = torch.tensor([0, 1],dtype=torch.float32)

    print(focal_loss_with_logits(x,y, alpha=a, gamma=g), binary_cross_entropy_with_logits(x,y))
