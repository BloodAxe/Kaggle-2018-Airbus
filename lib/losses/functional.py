import torch
import torch.nn.functional as F

alpha = 0.01
gamma = 1.5


def binary_focal_loss(x, y, alpha=0.25, gamma=2., reduction='none'):
    pt = x.detach() * (y.detach() * 2 - 1)
    w = (1 - pt).pow(gamma)
    w[y == 0] *= (1 - alpha)
    w[y > 0] *= alpha
    # a = torch.where(y < 0, alpha, (1 - alpha))
    loss = F.binary_cross_entropy(x, y, w, reduction=reduction)
    return loss


def binary_focal_loss_with_logits(x, y, alpha=0.25, gamma=2):
    '''Focal loss for binary case.
    Args:
      x: (tensor) predictions, sized [N,D].
      y: (tensor) targets, sized [N,].
    Return:
      (tensor) focal loss.
    '''

    # return binary_focal_loss(x.sigmoid(), y, alpha, gamma)

    y = y.view(x.size()).float()
    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    pt = F.logsigmoid(-x * (y * 2 - 1)).exp()
    w = pt.pow(gamma)
    w = torch.where(y < 0, w * alpha, w * (1 - alpha))
    loss = F.binary_cross_entropy_with_logits(x, y, w, reduction='none')
    return loss


def test_binary_focal_loss_with_logits():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(-4, 4, 0.1, dtype=np.float32)
    y = np.ones(len(x), dtype=np.float32)

    y_focal = binary_focal_loss_with_logits(torch.from_numpy(x), torch.from_numpy(y), alpha=alpha, gamma=gamma).numpy()
    y_bce = F.binary_cross_entropy_with_logits(torch.from_numpy(x), torch.from_numpy(y), reduction='none').numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y_focal, 'k--', label='Focal loss')
    ax.plot(x, y_bce, 'k:', label='BCE')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.title('test_binary_focal_loss_with_logits')
    plt.show()


def test_binary_focal_loss():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(0.0001, 1, 0.01, dtype=np.float32)
    y = np.ones(len(x), dtype=np.float32)

    y_focal = binary_focal_loss(torch.from_numpy(x), torch.from_numpy(y), alpha=alpha, gamma=gamma).numpy()
    y_bce = F.binary_cross_entropy(torch.from_numpy(x), torch.from_numpy(y), reduction='none').numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y_focal, 'k--', label='Focal loss')
    ax.plot(x, y_bce, 'k:', label='BCE')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.title('test_binary_focal_loss')
    plt.show()
