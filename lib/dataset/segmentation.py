import random

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, watershed, square, erosion, disk
from skimage.segmentation import find_boundaries

from lib.common.torch_utils import tensor_from_rgb_image
from lib.dataset.common import read_train_image, AirbusDataset, INDEX_KEY, ID_KEY, IMAGE_KEY, MASK_KEY


def instance_mask_to_fg_and_edge(mask):
    ships_mask = mask > 0
    all_boundaries = find_boundaries(mask)
    out_boundaries = find_boundaries(binary_fill_holes(ships_mask), mode='outter')

    touch_mask = all_boundaries.astype(np.uint8) - out_boundaries.astype(np.uint8)

    mask = np.dstack([ships_mask, touch_mask]).astype(np.uint8)
    return mask


def instance_mask_to_fg_and_edge_v2(labels):
    tmp = dilation(labels > 0, square(7))
    tmp2 = watershed(tmp, erosion(labels, disk(3)), mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(5))

    mask = np.dstack([labels > 0, tmp]).astype(np.uint8)
    return mask


def instance_mask_to_ships_mask(labels):
    labels = labels > 0
    return labels.astype(np.uint8).reshape(labels.shape[0], labels.shape[1], 1)


class SegmentationDataset(AirbusDataset):
    def __init__(self, sample_ids, data_dir, transform, groundtruth=None):
        super().__init__(sample_ids, data_dir, transform, groundtruth)

    def get_num_classes(self):
        """
        Mask has 2 channels - channel 0 indicates presence of ship, channel 1 indicates that pixel belongs to boundary between ship & non-ship
        :return:
        """
        return 2

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        id = self.sample_ids[index]
        image = read_train_image(id, self.data_dir)

        data = {
            INDEX_KEY: index,
            ID_KEY: id,
            IMAGE_KEY: image,
        }

        if self.groundtruth is not None:
            instance_mask = self.decode_mask(id)
            data[MASK_KEY] = instance_mask_to_fg_and_edge(instance_mask)

        data = self.transform(**data)
        data = self.normalize(**data)

        data[IMAGE_KEY] = tensor_from_rgb_image(data[IMAGE_KEY])

        image = data[IMAGE_KEY]

        if self.groundtruth is not None:
            data[MASK_KEY] = tensor_from_rgb_image(data[MASK_KEY]).float()

        y_true = {i: data[i] for i in data if i != IMAGE_KEY}
        return image, y_true


class OversampledSegmentationDataset(AirbusDataset):
    """
    This dataset implementation differs from SegmentationDataset by oversampling ships in crops
    """

    def __init__(self, sample_ids, data_dir, patch_size, transform, groundtruth):
        super().__init__(sample_ids, data_dir, transform, groundtruth)
        self.patch_size = patch_size

    def get_num_classes(self):
        """
        Mask has 2 channels - channel 0 indicates presence of ship, channel 1 indicates that pixel belongs to boundary between ship & non-ship
        :return:
        """
        return 2

    def __getitem__(self, index):

        id = self.sample_ids[index]

        data = {
            INDEX_KEY: index,
            ID_KEY: id,
            IMAGE_KEY: read_train_image(id, self.data_dir),
            MASK_KEY: self.decode_mask(id)
        }
        # print(data[IMAGE_KEY].shape, data[MASK_KEY].shape)

        if self.transform is not None:
            data = self.transform(**data)
        data = self.normalize(**data)

        # After augmentation let's apply oversampling by cropping ships
        if data[MASK_KEY].max() > 0:  # If we have ships
            ship_index = random.randint(1, data[MASK_KEY].max())
            ship_mask = (data[MASK_KEY] == ship_index).astype(np.uint8)
            rect = cv2.boundingRect(cv2.findNonZero(ship_mask))

            min_x = rect[0]
            min_y = rect[1]
            max_x = rect[2] + min_x
            max_y = rect[3] + min_y
        else:
            min_x = 0
            min_y = 0
            max_x = data[MASK_KEY].shape[1]
            max_y = data[MASK_KEY].shape[0]

        i = random.randint(min_y, max_y)
        j = random.randint(min_x, max_x)

        ic = np.clip(i - self.patch_size // 2, 0, data[MASK_KEY].shape[0] - self.patch_size - 1)
        jc = np.clip(j - self.patch_size // 2, 0, data[MASK_KEY].shape[1] - self.patch_size - 1)

        image = data[IMAGE_KEY][ic:ic + self.patch_size, jc:jc + self.patch_size]
        mask = data[MASK_KEY][ic:ic + self.patch_size, jc:jc + self.patch_size]

        if image.shape[0] == 0 or image.shape[1] == 0:
            breakpoint()

        image = tensor_from_rgb_image(image)
        data[IMAGE_KEY] = image

        mask = instance_mask_to_fg_and_edge(mask)
        mask = tensor_from_rgb_image(mask).float()
        data[MASK_KEY] = mask

        y_true = {i: data[i] for i in data if i != IMAGE_KEY}
        return image, y_true
