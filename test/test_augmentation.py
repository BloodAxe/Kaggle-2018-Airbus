import os
import pandas as pd
from albumentations import Compose, RandomRotate90, HorizontalFlip, VerticalFlip, RandomBrightness, IAAAdditiveGaussianNoise, MotionBlur, OneOf, ShiftScaleRotate, GaussNoise, MedianBlur, IAAPiecewiseAffine, \
    GridDistortion, Blur
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import make_grid

from dataset import default_transforms, test_augmentations
from models.common import find_in_dir

import matplotlib.pyplot as plt


def hard_augmentations(use_d4=False):
    d4 = Compose([
        RandomRotate90(p=1),
        HorizontalFlip(p=1),
        VerticalFlip(p=1)
    ])

    fliplr = HorizontalFlip()

    aug = Compose([
        RandomBrightness(p=1),
        OneOf([IAAAdditiveGaussianNoise(p=1), GaussNoise(p=1), ], p=0.5),
        OneOf([MotionBlur(p=1), MedianBlur(blur_limit=3, p=1), Blur(blur_limit=3, p=1)], p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=1, rotate_limit=5, p=1),
        OneOf([GridDistortion(p=1), IAAPiecewiseAffine(p=1)], p=0.5),
    ], p=1)

    default = default_transforms()

    return Compose([d4 if use_d4 else fliplr, aug, default])


def hard_augmentations2(use_d4=False):
    d4 = Compose([
        RandomRotate90(p=1),
        HorizontalFlip(p=1),
        VerticalFlip(p=1)
    ])

    fliplr = HorizontalFlip()

    aug = Compose([
        RandomBrightness(p=1),
        # OneOf([IAAAdditiveGaussianNoise(), , ], p=0.2),
        # GaussNoise(p=1),
        IAAAdditiveGaussianNoise(p=1),
        OneOf([MotionBlur(p=1), MedianBlur(blur_limit=3, p=1), Blur(blur_limit=3, p=1)], p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=1),
        # OneOf([GridDistortion(p=0.1), IAAPiecewiseAffine(p=0.3)], p=0.2),
        # GridDistortion(p=1),
        # IAAPiecewiseAffine(p=1)
    ], p=1)

    default = default_transforms()

    return Compose([d4 if use_d4 else fliplr, aug, default])


def profile_augmentations(data_dir='../data', use_d4=False, use_cumsum=False, use_depth=False, batch_size=32, workers=0):
    from dataset import ImageAndMaskDataset, select_depths
    from detect_bads import check_masks

    ids = [os.path.splitext(os.path.basename(fname))[0] for fname in find_in_dir(os.path.join(data_dir, 'train', 'images'))]

    # Drop train samples that has vertical strips
    all_black, all_white, vstrips = check_masks(find_in_dir(os.path.join(data_dir, 'train', 'masks')))
    ids = list(set(ids) - set(vstrips))

    images = [os.path.join(data_dir, 'train', 'images', '%s.png' % x) for x in ids]
    masks = [os.path.join(data_dir, 'train', 'masks', '%s.png' % x) for x in ids]

    depths = pd.read_csv(os.path.join(data_dir, 'depths.csv'))
    depths['z'] = depths['z'].astype(np.float32)
    depths['z'] = depths['z'] / depths['z'].max()
    depths = select_depths(depths, ids)

    train_transform = hard_augmentations2()

    trainset = ImageAndMaskDataset(images, masks, depths, transform=train_transform, append_depth=use_depth, append_cumsum=use_cumsum)
    num_channels = trainset.channels()

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             num_workers=workers,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True)

    for image, mask in trainloader:
        i = make_grid(image, normalize=True)
        i = i.detach().numpy()
        i = np.moveaxis(i, 0, -1)

        plt.figure()
        plt.imshow(i)
        plt.show()


if __name__ == '__main__':
    profile_augmentations()
