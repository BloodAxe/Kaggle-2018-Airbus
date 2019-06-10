import cv2
import numpy as np
import torch
from skimage.segmentation import relabel_sequential

from lib.common.rbox_util import from_opencv_rbox
from lib.common.torch_utils import tensor_from_rgb_image
from lib.dataset.common import read_train_image, ID_KEY, IMAGE_KEY, INDEX_KEY, SSD_LABELS_KEY, SSD_BBOXES_KEY, AirbusDataset, MASK_KEY, read_test_image


def instance_mask_to_rbboxes(instance_mask):
    """

    :param instance_mask:
    :return:
    """
    rects = []

    # Relabeling required becase small objects can disappear during augmentation / cropping
    instance_mask, _, _ = relabel_sequential(instance_mask)

    for i in range(1, instance_mask.max() + 1):
        points = cv2.findNonZero((instance_mask == i).astype(np.uint8))
        rbox = cv2.minAreaRect(points)
        rbox_new = from_opencv_rbox(rbox)
        rects.append(rbox_new)

    return np.array(rects, dtype=np.float32)


class RSSDDataset(AirbusDataset):
    def __init__(self, sample_ids, data_dir, transform, box_coder, groundtruth=None, test=False):
        super().__init__(sample_ids, data_dir, transform, groundtruth)
        self.box_coder = box_coder
        self.label_names = np.array(['Ship'])
        self.label_colors = np.array([(200, 0, 200)])
        self.test = test

    def get_num_classes(self):
        return 1

    def __getitem__(self, index):
        id = self.sample_ids[index]
        if self.test:
            image = read_test_image(id, self.data_dir)
        else:
            image = read_train_image(id, self.data_dir)

        data = {
            INDEX_KEY: index,
            ID_KEY: id,
            IMAGE_KEY: image,
        }

        if self.groundtruth is not None:
            data[MASK_KEY] = self.decode_mask(id)

        if self.transform is not None:
            data = self.transform(**data)

        data = self.normalize(**data)

        image = tensor_from_rgb_image(data[IMAGE_KEY])

        # Extract bboxes & convert them to RSSD format
        if self.groundtruth is not None:
            rbboxes = instance_mask_to_rbboxes(data[MASK_KEY])
            del data[MASK_KEY]

            labels = np.zeros(len(rbboxes), dtype=np.long)  # We have only one class -'ship' which is class 0
            rbboxes, labels = self.box_coder.encode(rbboxes, labels)
            data[SSD_BBOXES_KEY], data[SSD_LABELS_KEY] = torch.from_numpy(rbboxes), torch.from_numpy(labels).long()

        y_true = {i: data[i] for i in data if i != IMAGE_KEY}
        return image, y_true
