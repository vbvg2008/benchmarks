import pdb

import numpy as np
import torch
import torch.nn as nn


class HRnet(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    ResBlock(64, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64))
        self.stage2 = StageModule(num_module=1, num_block=(4, 4), c_in=(256, ), c_out=(32, 64))
        self.stage3 = StageModule(num_module=4, num_block=(4, 4, 4), c_in=(32, 64), c_out=(32, 64, 128))
        self.stage4 = StageModule(num_module=3, num_block=(4, 4, 4, 4), c_in=(32, 64, 128), c_out=(32, 64, 128, 256))
        self.final_layer = nn.Conv2d(32, num_classes, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2([x])  # starting from the second stage, x becomes a list
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_layer(x[0])  # for keypoint detection, only use the hightest resolution output
        x = torch.sigmoid(x)
        return x


class StageModule(nn.Module):
    def __init__(self, num_module, num_block, c_in, c_out):
        super().__init__()
        self.transition_module = self.make_transition(c_in, c_out)
        self.stage_module = nn.Sequential(*[HRModule(num_block, c_out) for _ in range(num_module)])

    def make_transition(self, c_in, c_out):
        num_branch_in, num_branch_out = len(c_in), len(c_out)
        transition_layers = []
        for idx in range(num_branch_out):
            if idx < num_branch_in:  # extending existing scale horizontally
                if c_in[idx] != c_out[idx]:
                    transition_layers.append(
                        nn.Sequential(nn.Conv2d(c_in[idx], c_out[idx], kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(c_out[idx]),
                                      nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:  # new vertical scale
                convs = []
                for j in range(idx + 1 - num_branch_in):
                    filter_in = c_in[-1]
                    filter_out = c_out[idx] if j == idx - num_branch_in else filter_in
                    convs.append(
                        nn.Sequential(nn.Conv2d(filter_in, filter_out, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(filter_out),
                                      nn.ReLU()))
                transition_layers.append(nn.Sequential(*convs))
        return nn.ModuleList(transition_layers)

    def forward(self, x):
        x = x + [x[-1] for _ in range(len(self.transition_module) - len(x))]  # make sure the x is padded with x[-1]
        x = [m(data) if m is not None else data for m, data in zip(self.transition_module, x)]
        x = self.stage_module(x)
        return x


class HRModule(nn.Module):
    def __init__(self, num_conv, c_out):
        super().__init__()
        self.branches = nn.ModuleList([self._make_branch(nconv, plane) for nconv, plane in zip(num_conv, c_out)])
        self.fuse_layers = self._make_fuse_layers(c_out)
        self.relu = nn.ReLU()

    def _make_branch(self, conv_count, plane):
        return nn.Sequential(*[BasicBlock(plane, plane) for _ in range(conv_count)])

    def _make_fuse_layers(self, channels):
        fuse_layers = []
        # length of c_out means number of branches
        for idx_out, planes_out in enumerate(channels):
            fuse_layer = []
            for idx_in, planes_in in enumerate(channels):
                if idx_in == idx_out:
                    fuse_layer.append(None)
                elif idx_in > idx_out:  # low-res -> high-res, need upsampling
                    fuse_layer.append(
                        nn.Sequential(nn.Conv2d(planes_in, planes_out, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(planes_out),
                                      nn.Upsample(scale_factor=2**(idx_in - idx_out), mode='nearest')))
                else:  # high-res -> low-res, need strided conv
                    convs = [
                        nn.Sequential(nn.Conv2d(planes_in, planes_in, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(planes_in),
                                      nn.ReLU()) for _ in range(idx_out - idx_in - 1)
                    ] + [
                        nn.Sequential(nn.Conv2d(planes_in, planes_out, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(planes_out))
                    ]
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        x = [branch(x_i) for branch, x_i in zip(self.branches, x)]
        x = [
            sum([x_i if layer is None else layer(x_i) for layer, x_i in zip(fuse_layers, x)])
            for fuse_layers in self.fuse_layers
        ]
        x = [self.relu(x_i) for x_i in x]
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        # when shape mismatch between input and conv output, apply input_conv to input
        if inplanes != planes * 4 or stride != 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        else:
            self.input_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.input_conv is not None:
            residual = self.input_conv(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        # when shape mismatch between input and conv output, apply input_conv to input
        if inplanes != planes or stride != 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.input_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.input_conv is not None:
            residual = self.input_conv(x)
        out += residual
        out = self.relu(out)
        return out


if __name__ == "__main__":
    model = HRnet()
    data = torch.Tensor(np.float32(np.random.rand(1, 3, 256, 256)))
    result = model(data)
    pdb.set_trace()
