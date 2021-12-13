import pdb

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.normalization import GroupNorm


def pad_with_coord(data):
    batch_size, _, height, width = data.shape
    x = torch.linspace(-1, 1, width, dtype=data.dtype).view(1, 1, 1, -1).expand(batch_size, 1, height, width)
    y = torch.linspace(-1, 1, height, dtype=data.dtype).view(1, 1, -1, 1).expand(batch_size, 1, height, width)
    data = torch.cat([data, x, y], axis=1)  # concatenate along channel dimension
    return data


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_C5 = nn.Conv2d(2048, 256, 1)
        self.conv2d_C4 = nn.Conv2d(1024, 256, 1)
        self.conv2d_C3 = nn.Conv2d(512, 256, 1)
        self.conv2d_C2 = nn.Conv2d(256, 256, 1)
        self.conv2d_P5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2d_P2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, C2, C3, C4, C5):
        # lateral conv
        P5 = self.conv2d_C5(C5)
        P5_up = nn.functional.interpolate(P5, scale_factor=2)
        P4 = self.conv2d_C4(C4)
        P4 = P4 + P5_up
        P4_up = nn.functional.interpolate(P4, scale_factor=2)
        P3 = self.conv2d_C3(C3)
        P3 = P3 + P4_up
        P3_up = nn.functional.interpolate(P3, scale_factor=2)
        P2 = self.conv2d_C2(C2)
        P2 = P2 + P3_up
        # fpn conv
        P5 = self.conv2d_P5(P5)
        P4 = self.conv2d_P4(P4)
        P3 = self.conv2d_P3(P3)
        P2 = self.conv2d_P2(P2)
        return P2, P3, P4, P5


class ConvNorm(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False)
        self.groupnorm = nn.GroupNorm(num_groups=groups, num_channels=c_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        return x


class MaskHead(nn.Module):
    def __init__(self, mid_ch=128, out_ch=256):
        super().__init__()
        self.convnorm_p2 = ConvNorm(256, mid_ch)
        self.convnorm_p3 = ConvNorm(256, mid_ch)
        self.convnorm_p4a = ConvNorm(256, mid_ch)
        self.convnorm_p4b = ConvNorm(mid_ch, mid_ch)
        self.convnorm_p5a = ConvNorm(258, mid_ch)
        self.convnorm_p5b = ConvNorm(mid_ch, mid_ch)
        self.convnorm_p5c = ConvNorm(mid_ch, mid_ch)
        self.convnorm_out = ConvNorm(mid_ch, out_ch, kernel_size=1)

    def forward(self, P2, P3, P4, P5):
        # first level
        P2 = nn.functional.relu(self.convnorm_p2(P2))
        # second level
        P3 = nn.functional.relu(self.convnorm_p3(P3))
        P3 = nn.functional.interpolate(P3, scale_factor=2)
        # third level
        P4 = nn.functional.relu(self.convnorm_p4a(P4))
        P4 = nn.functional.interpolate(P4, scale_factor=2)
        P4 = nn.functional.relu(self.convnorm_p4b(P4))
        P4 = nn.functional.interpolate(P4, scale_factor=2)
        # top level, add coordinate
        P5 = nn.functional.relu(self.convnorm_p5a(pad_with_coord(P5)))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        P5 = nn.functional.relu(self.convnorm_p5b(P5))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        P5 = nn.functional.relu(self.convnorm_p5c(P5))
        P5 = nn.functional.interpolate(P5, scale_factor=2)
        seg_outputs = nn.functional.relu(self.convnorm_out(P2 + P3 + P4 + P5))
        return seg_outputs


class HeadModel(nn.Module):
    def __init__(self, num_classes=80, chin_in=258, ch_feature=512, ch_kernel_out=256):
        super().__init__()
        self.convnorm_k1 = ConvNorm(chin_in, ch_feature)
        self.convnorm_c1 = ConvNorm(chin_in - 2, ch_feature)
        self.convnorm_k2 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c2 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_k3 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c3 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_k4 = ConvNorm(ch_feature, ch_feature)
        self.convnorm_c4 = ConvNorm(ch_feature, ch_feature)
        self.conv_kernel = nn.Conv2d(ch_feature, ch_kernel_out, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_kernel.weight.data, std=0.01)
        nn.init.zeros_(self.conv_kernel.bias.data)
        self.conv_cls = nn.Conv2d(ch_feature, num_classes, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_cls.weight.data, std=0.01)
        nn.init.constant_(self.conv_cls.bias.data, val=np.log(1 / 99))

    def forward(self, x):
        feature_kernel = x
        feature_cls = x[:, :-2, :, :]
        feature_kernel = nn.functional.relu(self.convnorm_k1(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c1(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k2(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c2(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k3(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c3(feature_cls))
        feature_kernel = nn.functional.relu(self.convnorm_k4(feature_kernel))
        feature_cls = nn.functional.relu(self.convnorm_c4(feature_cls))
        feature_kernel = self.conv_kernel(feature_kernel)
        feature_cls = self.conv_cls(feature_cls)
        return feature_kernel, feature_cls


class SoloV2Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool_p2 = nn.MaxPool2d(kernel_size=2)
        self.head_model = HeadModel(num_classes=num_classes)

    def forward(self, P2, P3, P4, P5):
        # applying maxpool first for P2
        P2 = self.maxpool_p2(P2)
        features = [P2, P3, P4, P5, P5]
        grid_sizes = [40, 36, 24, 16, 12]
        feat_kernel_list, feat_cls_list = [], []
        for feature, grid_size in zip(features, grid_sizes):
            feature = pad_with_coord(feature)
            feature = nn.functional.interpolate(feature,
                                                size=(grid_size, grid_size),
                                                mode='bilinear',
                                                align_corners=False)
            feat_kernel, feat_cls = self.head_model(feature)
            feat_kernel_list.append(feat_kernel)
            feat_cls_list.append(feat_cls)
        return feat_cls_list, feat_kernel_list


class SoloV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        res50_layers = list(torchvision.models.resnet50(pretrained=True).children())
        self.res50_in_C2 = nn.Sequential(*(res50_layers[:5]))
        self.res50_C2_C3 = nn.Sequential(*(res50_layers[5]))
        self.res50_C3_C4 = nn.Sequential(*(res50_layers[6]))
        self.res50_C4_C5 = nn.Sequential(*(res50_layers[7]))
        self.fpn = FPN()
        self.mask_head = MaskHead()
        self.head = SoloV2Head(num_classes=num_classes)

    def forward(self, x):
        C2 = self.res50_in_C2(x)
        C3 = self.res50_C2_C3(C2)
        C4 = self.res50_C3_C4(C3)
        C5 = self.res50_C4_C5(C4)
        P2, P3, P4, P5 = self.fpn(C2, C3, C4, C5)
        feat_seg = self.mask_head(P2, P3, P4, P5)
        feat_cls_list, feat_kernel_list = self.head(P2, P3, P4, P5)
        return feat_seg, feat_cls_list, feat_kernel_list


if __name__ == "__main__":
    data = np.random.rand(8, 3, 1024, 1024)
    data = torch.Tensor(data)
    # data = nn.functional.interpolate(data, size=(30, 30))
    # data = pad_with_coord(data)
    model = SoloV2(80)
    feat_seg, feat_cls_list, feat_kernel_list = model(data)
    pdb.set_trace()
