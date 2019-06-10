import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.common.fs import find_in_dir

INDEX_KEY = 'index'
ID_KEY = 'id'
IMAGE_KEY = 'image'
BBOXES_KEY = 'bboxes'
LABELS_KEY = 'labels'
MASK_KEY = 'mask'
SSD_BBOXES_KEY = 'ssd_bboxes'
SSD_LABELS_KEY = 'ssd_labels'
SHIP_PRESENSE_KEY = 'has_ship'

def read_train_image(id, data_dir):
    fname = os.path.join(data_dir, 'train_v2', f'{id}')
    image = cv2.imread(fname)
    if image is None:
        raise ValueError(f'Cannot read image {fname}')
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def read_test_image(id, data_dir):
    fname = os.path.join(data_dir, 'test_v2', f'{id}')
    image = cv2.imread(fname)
    if image is None:
        raise ValueError(f'Cannot read image {fname}')
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def rle_decode(mask_rle, shape=(768, 768), value=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    if type(mask_rle) == float:
        return img.reshape(shape).T

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = value
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2instancemask(rles, shape=(768, 768)):
    """
    Decodes list of rle masks into label image
    :param rles:
    :param shape:
    :return:
    """
    mask = np.zeros(shape, dtype=np.uint16)

    for index, rle in enumerate(rles):
        if isinstance(rle, str):
            mask += rle_decode(rle, shape, index + 1)
    return mask


def all_train_ids(data_dir) -> np.ndarray:
    """
    Return all train ids
    :return: Numpy array of ids
    """
    return np.array(
        sorted([os.path.basename(fname).lower() for fname in find_in_dir(os.path.join(data_dir, 'train_v2'))]))


def all_test_ids(data_dir) -> np.ndarray:
    """
    Return all train ids
    :return: Numpy array of ids
    """
    return np.array(
        sorted([os.path.basename(fname).lower() for fname in find_in_dir(os.path.join(data_dir, 'test_v2'))]))


def get_train_test_split_for_fold(fold: int, ships_only=False):
    if ships_only:
        split_ds = 'folds_nonempty.csv'
    else:
        split_ds = 'folds.csv'

    split_ds = pd.read_csv(os.path.join('lib', 'dataset', split_ds))
    train_ids = np.array(split_ds[split_ds['Fold'] != fold].ImageId)
    valid_ids = np.array(split_ds[split_ds['Fold'] == fold].ImageId)

    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError('Detected duplicate entries in train_ids')

    if len(valid_ids) != len(set(valid_ids)):
        raise RuntimeError('Detected duplicate entries in valid_ids')

    return train_ids, valid_ids


def get_rle_from_groundtruth(groundtruth, id):
    view = groundtruth[groundtruth['ImageId'] == id]
    encoded_pixels = view['EncodedPixels']
    return encoded_pixels


def get_bboxes_from_groundtruth(groundtruth):
    bboxes = []
    for i, row in tqdm(groundtruth.iterrows(), total=len(groundtruth)):
        encoded_pixels = row['EncodedPixels']
        mask = rle_decode(encoded_pixels)
        if mask.max() > 0:
            pts = cv2.findNonZero(mask)
            rect = cv2.boundingRect(pts)
        else:
            rect = [0, 0, 0, 0]

        bboxes.append(rect)
    return np.array(bboxes)


class AirbusDataset(Dataset):
    def __init__(self, sample_ids, data_dir, transform, groundtruth=None, normalize_op=A.Normalize()):
        self.data_dir = data_dir
        self.sample_ids = np.array(sample_ids)
        self.transform = transform
        self.normalize = normalize_op
        self.groundtruth = groundtruth

    def __len__(self):
        return len(self.sample_ids)

    def decode_mask(self, id):
        rle = get_rle_from_groundtruth(self.groundtruth, id)
        return rle2instancemask(rle)

    def __getitem__(self, item):
        raise NotImplementedError


def get_transform(training, width=512, height=512):
    t = A.Compose([
        A.Resize(width, height),
        # Augmentation
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.JpegCompression(quality_lower=90),
            A.RandomBrightness(),
            A.RandomContrast(),
            # A.RandomGamma(),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20),
            A.CLAHE(),
            A.MedianBlur(),
            A.MotionBlur(),
            A.GaussNoise()
        ], p=float(training)),
    ])

    return t
