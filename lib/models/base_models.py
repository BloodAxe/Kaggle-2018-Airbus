import albumentations as A
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common.torch_utils import to_numpy, tensor_from_rgb_image
from lib.dataset.common import ID_KEY, rle_encode
from lib.tiles import ImageSlicer
from lib.visualizations import visualize_rbbox


class EncoderModule(nn.Module):

    def forward(self, image):
        features = []
        x = image
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return features

    @property
    def encoder_layers(self):
        raise NotImplementedError

    @property
    def output_strides(self):
        raise NotImplementedError

    @property
    def output_filters(self):
        raise NotImplementedError


class BaseDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()

    def predict_as_csv(self, dataset, batch_size=1, workers=0):
        self.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

        image_ids = []
        rles = []

        for image, y in tqdm(dataloader, total=len(dataloader)):
            image = image.cuda(non_blocking=True)
            pred_boxes, pred_labels = self(image)
            for image_id, boxes, scores in zip(y[ID_KEY], to_numpy(pred_boxes), to_numpy(pred_labels)):
                boxes, _, scores = self.box_coder.decode(boxes, scores)
                if len(boxes):
                    for rbox in boxes:
                        mask = np.zeros((768, 768), dtype=np.uint8)
                        visualize_rbbox(mask, rbox, color=(1, 1, 1), thickness=cv2.FILLED)
                        rle = rle_encode(mask)

                        image_ids.append(image_id)
                        rles.append(rle)
                else:
                    image_ids.append(image_id)
                    rles.append(None)

        return pd.DataFrame.from_dict({'ImageId': image_ids, 'EncodedPixels': rles})


class BaseSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

    def predict(self, image, patch_size, batch_size=1, normalize=A.Normalize()):
        self.eval()

        slicer = ImageSlicer(image.shape, patch_size, patch_size // 2)
        image = normalize(image=image)['image']
        tiles = [tensor_from_rgb_image(tile) for tile in slicer.split(image)]

        loader = DataLoader(tiles, batch_size=batch_size, pin_memory=True)
        pred_tiles = []
        for batch in loader:
            batch = batch.cuda(non_blocking=True)
            pred = self(batch)
            pred = to_numpy(pred)
            pred = np.moveaxis(pred, 1, -1)
            pred_tiles.extend(pred)

        mask = slicer.merge(pred_tiles)
        return mask
