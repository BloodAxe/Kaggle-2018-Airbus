import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from lib.common.torch_utils import tensor_from_rgb_image, to_numpy, count_parameters
from lib.models.fpnssd512.box_coder import FPNSSDBoxCoder
from lib.models.mobilenetssd512.mobilenet import MobileNetV2
from lib.tiles import ImageSlicer


class MobilenetSSD512(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes, **kwargs):
        super(MobilenetSSD512, self).__init__()
        encoder = MobileNetV2()

        self.layer0 = encoder.layer0
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.layer5 = encoder.layer5
        self.layer6 = encoder.layer6
        self.layer7 = encoder.layer7

        self.conv6 = nn.Conv2d(320, 64, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.num_classes = num_classes + 1  # Dummy class
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.box_coder = FPNSSDBoxCoder()

    def forward(self, image):

        # Extract features
        c0 = self.layer0(image)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        c6 = self.layer6(c5)
        c7 = self.layer7(c6)

        # print(c0.size())
        # print(c2.size())
        # print(c3.size())
        # print(c4.size())
        # print(c5.size())
        # print(c6.size())
        # print(c7.size())

        p6 = self.conv6(c7)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))

        # Top-down
        p5 = self.toplayer(c7)
        p4 = self._upsample_add(p5, self.latlayer1(c5))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)

        features = [p3, p4, p5, p6, p7, p8, p9]

        loc_preds = []
        cls_preds = []

        for fm in features:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1, 4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1, self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        bboxes = torch.cat(loc_preds, 1)
        labels = torch.cat(cls_preds, 1)

        return bboxes, labels

    def _upsample_add(self, x, y, scale_factor=2):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # print(x.size(), y.size())
        return F.interpolate(x, scale_factor=2, mode='nearest') + y

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(64, out_planes, kernel_size=3, stride=1, padding=1))
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
                patch = patch.to(self.conv6.weight.device)
                bboxes, labels = self(patch)

                all_bboxes.extend(bboxes.cpu())
                all_labels.extend(labels.cpu())

        boxes, labels, scores = self.box_coder.decode_multi(all_bboxes, all_labels, offsets)
        return to_numpy(boxes), to_numpy(labels), to_numpy(scores)


def test():
    model = MobilenetSSD512(1)
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
