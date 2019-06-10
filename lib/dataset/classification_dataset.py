import torch
import numpy as np
from lib.common.torch_utils import tensor_from_rgb_image
from lib.dataset.common import read_train_image, AirbusDataset, INDEX_KEY, ID_KEY, IMAGE_KEY, MASK_KEY, SHIP_PRESENSE_KEY


class ClassificationDataset(AirbusDataset):
    def __init__(self, sample_ids, data_dir, transform, groundtruth=None):
        super().__init__(sample_ids, data_dir, transform, groundtruth)

    def get_num_classes(self):
        """
        We predict whether there are any ships on mask or not
        :return:
        """
        return 1

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
            data[MASK_KEY] = self.decode_mask(id)

        if self.transform is not None:
            data = self.transform(**data)

        data = self.normalize(**data)
        data[IMAGE_KEY] = tensor_from_rgb_image(data[IMAGE_KEY])

        image = data[IMAGE_KEY]

        if self.groundtruth is not None:
            data[SHIP_PRESENSE_KEY] = np.array([(data[MASK_KEY] > 0).any()], dtype=np.float32)
            data[MASK_KEY] = data[MASK_KEY].astype(int)
            y_true = {i: data[i] for i in data if i != IMAGE_KEY}
            return image, y_true
        else:
            return image
