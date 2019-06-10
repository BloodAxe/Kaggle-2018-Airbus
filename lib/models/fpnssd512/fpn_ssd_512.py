import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from lib.common.torch_utils import tensor_from_rgb_image, to_numpy, count_parameters
from lib.models.fpnssd512.box_coder import FPNSSDBoxCoder
from lib.models.fpnssd512.fpn import FPN50
from lib.tiles import ImageSlicer


class FPNSSD512(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(FPNSSD512, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes + 1  # Dummy class
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.box_coder = FPNSSDBoxCoder()

        resnet_state = resnet50(pretrained=pretrained).state_dict()
        self.fpn.load_state_dict(resnet_state, strict=False)
        # new_state_dict = OrderedDict()
        # for k, v in resnet_state.items():
        #     if str.startswith(k, 'conv1'):
        #         continue
        #     new_state_dict[k] = v

    def forward(self, image):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(image)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1, 4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1, self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        bboxes = torch.cat(loc_preds, 1)
        labels = torch.cat(cls_preds, 1)

        return bboxes, labels
        # return {
        #     SSD_BBOXES_KEY: bboxes,
        #     SSD_LABELS_KEY: labels,
        # }

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def predict(self, image):
        import albumentations as A
        self.eval()

        normalize = A.Normalize()
        image = normalize(image=image)['image']

        slicer = ImageSlicer(image.shape, 512, 512 // 2)
        patches = [tensor_from_rgb_image(patch) for patch in slicer.split(image, borderType=cv2.BORDER_CONSTANT)]
        offsets = torch.tensor([[crop[0], crop[1], crop[0], crop[1]] for crop in slicer.bbox_crops], dtype=torch.float32)

        all_bboxes = []
        all_labels = []

        with torch.set_grad_enabled(False):
            for patch, patch_loc in DataLoader(list(zip(patches, offsets)), batch_size=8, pin_memory=True):
                patch = patch.to(self.fpn.conv1.weight.device)
                bboxes, labels = self(patch)

                all_bboxes.extend(bboxes.cpu())
                all_labels.extend(labels.cpu())

        boxes, labels, scores = self.box_coder.decode_multi(all_bboxes, all_labels, offsets)
        return to_numpy(boxes), to_numpy(labels), to_numpy(scores)


def test():
    model = FPNSSD512(1)
    image = torch.randn(4, 3, 512, 512)
    output = model(image)
    print(count_parameters(model))

    model.cuda()
    image = image.cuda()

    with torch.cuda.profiler.profile() as prof:
        model(image)  # Warmup CUDA memory allocator and profiler
        with torch.autograd.profiler.emit_nvtx():
            model(image)

    print(prof)


if __name__ == '__main__':
    test()
