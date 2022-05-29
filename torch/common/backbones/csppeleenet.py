#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyTorch implementation of CSPPeleeNet,
# ported from https://github.com/tomotana14/CSPPeleeNet.pytorch
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class Conv3x3(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv3x3, self).__init__(
            nn.Conv2d(inp, oup, 3, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class Conv1x1(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv1x1, self).__init__(
            nn.Conv2d(inp, oup, 1, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class StemBlock(nn.Module):
    def __init__(self, inp):
        super(StemBlock, self).__init__()
        self.conv1 = Conv3x3(inp, 32, 2, 1)
        self.conv2 = nn.Sequential(
            Conv1x1(32, 16, 1, 0),
            Conv3x3(16, 32, 2, 1)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv1x1(64, 32, 1, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat_ = self.conv2(feat)
        feat = torch.cat([self.max_pool(feat), feat_], dim=1)
        feat = self.conv3(feat)
        return feat


class TwoWayDenseBlock(nn.Module):
    def __init__(self, inp, growth_rate, inter_ch):
        super(TwoWayDenseBlock, self).__init__()
        self.left = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1)
        )
        self.right = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1),
            Conv3x3(growth_rate//2, growth_rate//2, 1, 1)
        )

    def forward(self, x):
        feat_l = self.left(x)
        feat_r = self.right(x)
        feat = torch.cat([x, feat_l, feat_r], dim=1)
        return feat


class TransitionBlock(nn.Sequential):
    def __init__(self, inp, pool=True):
        if pool:
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            pool = nn.Identity()
        super(TransitionBlock, self).__init__(
            Conv1x1(inp, inp, 1, 0),
            pool
        )


class DenseStage(nn.Module):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool):
        super(DenseStage, self).__init__()
        current_ch = inp
        inter_ch = int(growth_rate // 2 * bwidth / 4) * 4
        dense_branch = nn.Sequential()
        for i in range(nblock):
            dense_branch.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate, inter_ch))
            current_ch += growth_rate
        dense_branch.add_module(
            "transition1", TransitionBlock(current_ch, pool=pool))
        self.dense_branch = dense_branch

    def forward(self, x):
        return self.dense_branch(x)


class CSPDenseStage(DenseStage):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool, partial_ratio):
        split_ch = int(inp * partial_ratio)
        super(CSPDenseStage, self).__init__(
            split_ch, nblock, bwidth, growth_rate, False)
        self.split_ch = split_ch
        current_ch = inp + (growth_rate * nblock)
        self.transition2 = TransitionBlock(current_ch, pool=pool)

    def forward(self, x):
        x1 = x[:, :self.split_ch, ...]
        x2 = x[:, self.split_ch:, ...]
        feat1 = self.dense_branch(x1)
        feat = torch.cat([x2, feat1], dim=1)
        feat = self.transition2(feat)
        return feat


class CSPPeleeNet(nn.Module):
    def __init__(self, inp=3, nclass=1000, growth_rate=32, nblocks=[3, 4, 8, 6],
                 bottleneck_widths=[1, 2, 4, 4], partial_ratio=0.5):
        super(CSPPeleeNet, self).__init__()

        self.stem = StemBlock(inp)
        current_ch = 32
        stages = nn.Sequential()
        pool = True
        assert len(nblocks) == len(bottleneck_widths)
        for i, (nblock, bwidth) in enumerate(zip(nblocks, bottleneck_widths)):
            if (i+1) == len(nblocks):
                pool = False
            if partial_ratio < 1.0:
                stage = CSPDenseStage(
                    current_ch, nblock, bwidth, growth_rate, pool, partial_ratio)
            else:
                stage = DenseStage(current_ch, nblock,
                                   bwidth, growth_rate, pool)
            stages.add_module("stage{}".format(i+1), stage)
            current_ch += growth_rate * nblock
        self.stages = stages
        #self.features = nn.Sequential(OrderedDict([
            #('stem', self.stem),
            #('stages', self.stages),
        #]))
        self.classifier = nn.Linear(current_ch, nclass)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.stages(feat)
        #feat = torch.mean(feat, dim=[2, 3])  # GAP
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.view(-1, feat.size(1))
        pred = self.classifier(feat)
        return pred


def csppeleenet(**kwargs):
    """
    Constructs a CSPPeleeNet model
    """
    if 'pretrained' in kwargs.keys():
        pretrained = kwargs['pretrained']
        kwargs.pop('pretrained')
    else:
        pretrained = None

    if 'weights_path' in kwargs.keys():
        weights_path = kwargs['weights_path']
        kwargs.pop('weights_path')
    else:
        weights_path = None

    model = CSPPeleeNet(**kwargs)

    if pretrained:
        # load pretrained weights
        weights_url = 'https://github.com/tomotana14/CSPPeleeNet.pytorch/raw/master/weights/csppeleenet.pth'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrain_dict = model_zoo.load_url(weights_url, map_location=device)
        model.load_state_dict(pretrain_dict)

    if weights_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model


if __name__ == '__main__':
    import os, sys
    from PIL import Image
    import torch.nn.functional as F
    from torchsummary import summary
    from tensorflow.keras.applications.resnet50 import decode_predictions

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
    from common.data_utils import preprocess_image

    model = csppeleenet(partial_ratio=0.5, pretrained=True)
    summary(model, input_size=(3, 224, 224))

    input_tensor = torch.randn(1, 3, 224, 224)
    #import torch.onnx
    #torch.onnx.export(model, input_tensor, "csppeleenet50.onnx", verbose=False)
    from thop import profile, clever_format
    macs, params = profile(model, inputs=(input_tensor, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('Total MACs: {}'.format(macs))
    print('Total PARAMs: {}'.format(params))

    #torch.save(model, 'check.pth')

    model.eval()
    with torch.no_grad():
        # prepare input image
        image = Image.open('../../../example/grace_hopper.jpg').convert('RGB')
        image_data = preprocess_image(image, target_size=(224, 224), return_tensor=True)
        image_data = image_data.unsqueeze(0)

        preds = model(image_data)
        preds = F.softmax(preds, dim=1).cpu().numpy()

    print('Predicted:', decode_predictions(preds))
