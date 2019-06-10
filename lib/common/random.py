import random
import torch


def set_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
