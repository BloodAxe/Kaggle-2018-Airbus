import os

import numpy as np
import pandas as pd
from skimage.color import label2rgb
from tqdm import tqdm

from lib.dataset.common import get_bboxes_from_groundtruth, rle2instancemask


def extract_masks(df, destination_dir):
    import matplotlib.pyplot as plt

    ids = np.unique(df['ImageId'])

    os.makedirs(destination_dir, exist_ok=True)
    for image_id in tqdm(ids, total=len(ids)):
        rles = df.loc[df['ImageId'] == image_id, 'EncodedPixels'].values
        mask = rle2instancemask(rles)
        np.save(os.path.join(destination_dir, '%s.npy' % image_id), mask)
        plt.imshow(label2rgb(mask, bg_label=0))


def main():
    data_dir = 'D:\\datasets\\airbus'
    train_ship_segmentation = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))

    # print(compute_mean_std(find_in_dir(os.path.join(data_dir,'data/train/images')) + find_in_dir(os.path.join(data_dir,'data/test/images'))))
    # extract_masks(train_ship_segmentation, os.path.join(data_dir, 'train_masks'))

    bboxes = get_bboxes_from_groundtruth(train_ship_segmentation)
    train_ship_segmentation['bbox_x'] = bboxes[:, 0]
    train_ship_segmentation['bbox_y'] = bboxes[:, 1]
    train_ship_segmentation['bbox_w'] = bboxes[:, 2]
    train_ship_segmentation['bbox_h'] = bboxes[:, 3]
    train_ship_segmentation.to_csv('data/train_ship_segmentations_w_bbox.csv', index=None)


if __name__ == '__main__':
    main()
