#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a HBONet Model as defined in:
HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions
https://arxiv.org/abs/1908.03888
Ported from https://github.com/d-li14/HBONet
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['hbonet']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, activation=True):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True) if activation else nn.Sequential()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class HarmoniousBottleneck_4x(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(HarmoniousBottleneck_4x, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.oup = oup

        self.conv = nn.Sequential(
            # dw-linear
            nn.Conv2d(inp, inp, 5, 2, 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw-linear
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 5, 2, 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # pw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, 1, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            #nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
        )

        if self.stride == 1:
            self.upsample = nn.Upsample(scale_factor=4)
        elif self.stride == 2:
            self.upsample = nn.Upsample(scale_factor=2)
            self.avgpool = nn.AvgPool2d(kernel_size=2)

        # dw-linear
        self.upconv = nn.Sequential(
            nn.Conv2d(oup // 2, oup // 2, 5, 1, 2, groups=oup // 2, bias=False),
            nn.BatchNorm2d(oup // 2),
        )

    def forward(self, x):
        if self.stride == 1:
            return torch.cat((x[:, -(self.oup - self.oup // 2):, :, :], \
                              self.upconv(x[:, :(self.oup // 2), :, :] + self.upsample(self.conv(x)))), dim=1)
        elif self.stride == 2:
            return torch.cat((self.avgpool(x[:, -(self.oup - self.oup // 2):, :, :]), \
                              self.upconv(self.avgpool(x[:, :(self.oup // 2), :, :]) + self.upsample(self.conv(x)))), dim=1)


class HarmoniousBottleneck_8x(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(HarmoniousBottleneck_8x, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.oup = oup

        self.conv = nn.Sequential(
            # dw-linear
            nn.Conv2d(inp, inp, 5, 2, 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw-linear
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 5, 2, 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # pw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw-linear
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 5, 2, 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # pw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, 1, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            #nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
        )

        if self.stride == 1:
            self.upsample = nn.Upsample(scale_factor=8)
        elif self.stride == 2:
            self.upsample = nn.Upsample(scale_factor=4)
            self.avgpool = nn.AvgPool2d(kernel_size=2)

        # dw-linear
        self.upconv = nn.Sequential(
            nn.Conv2d(oup // 2, oup // 2, 5, 1, 2, groups=oup // 2, bias=False),
            nn.BatchNorm2d(oup // 2),
        )

    def forward(self, x):
        if self.stride == 1:
            return torch.cat((x[:, -(self.oup - self.oup // 2):, :, :], \
                              self.upconv(x[:, :(self.oup // 2), :, :] + self.upsample(self.conv(x)))), dim=1)
        elif self.stride == 2:
            return torch.cat((self.avgpool(x[:, -(self.oup - self.oup // 2):, :, :]), \
                              self.upconv(self.avgpool(x[:, :(self.oup // 2), :, :]) + self.upsample(self.conv(x)))), dim=1)


class HarmoniousBottleneck_2x(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(HarmoniousBottleneck_2x, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.oup = oup

        self.conv = nn.Sequential(
            # dw-linear
            nn.Conv2d(inp, inp, 5, 2, 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, 1, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
        )

        if self.stride == 1:
            self.upsample = nn.Upsample(scale_factor=2)

            # dw-linear
            self.upconv = nn.Sequential(
                nn.Conv2d(oup // 2, oup // 2, 5, 1, 2, groups=oup // 2, bias=False),
                nn.BatchNorm2d(oup // 2),
            )
        elif self.stride == 2:
            self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        if self.stride == 1:
            return torch.cat((x[:, -(self.oup - self.oup // 2):, :, :], \
                              self.upconv(x[:, :(self.oup // 2), :, :] + self.upsample(self.conv(x)))), dim=1)
        elif self.stride == 2:
            return torch.cat((self.avgpool(x[:, -(self.oup - self.oup // 2):, :, :]), self.conv(x)), dim=1)


class HBONet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(HBONet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s, block
            [1,  20, 1, 1,        InvertedResidual],

            # alternative blocks for 8x varaint model
            # [2,  36, 1, 1, HarmoniousBottleneck_8x],
            # [2,  72, 3, 2, HarmoniousBottleneck_8x],
            # [2,  96, 4, 2, HarmoniousBottleneck_4x],

            # alternative blocks for 4x varaint model
            # [2,  36, 1, 1, HarmoniousBottleneck_4x],
            # [2,  72, 3, 2, HarmoniousBottleneck_4x],
            # [2,  96, 4, 2, HarmoniousBottleneck_4x],

            # alternative blocks for 2x main model
            [2,  36, 1, 1, HarmoniousBottleneck_2x],
            [2,  72, 3, 2, HarmoniousBottleneck_2x],
            [2,  96, 4, 2, HarmoniousBottleneck_2x],

            # fixed blocks
            [2, 192, 4, 2, HarmoniousBottleneck_2x],
            [2, 288, 1, 1, HarmoniousBottleneck_2x],
            [0, 144, 1, 1,             conv_1x1_bn],
            [6, 200, 2, 2,        InvertedResidual],
            [6, 400, 1, 1,        InvertedResidual],
        ]

        if width_mult == 0.1:
            divisible_value = 4
        # comment the following two lines to revert to 8 for variant models
        elif width_mult == 0.25:
            divisible_value = 2
        # comment the following two lines to revert to 8 for variant models
        elif width_mult == 0.5:
            divisible_value = 2
        else:
            divisible_value = 8
        # building first layer
        input_channel = _make_divisible(32 * width_mult, divisible_value)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, block in self.cfgs:
            output_channel = _make_divisible(c * width_mult, divisible_value)
            if block is not conv_1x1_bn:
                for i in range(n):
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                    input_channel = output_channel
            else:
                layers.append(conv_1x1_bn(input_channel, output_channel, activation=False))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1600 * width_mult, divisible_value) if width_mult > 1.0 else 1600
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def hbonet(**kwargs):
    """
    Constructs a HBONet model
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

    if 'width_mult' in kwargs.keys():
        width_mult = kwargs['width_mult']
    else:
        width_mult = 1.

    model = HBONet(**kwargs)

    if pretrained:
        # load pretrained weights
        if width_mult == 1.:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_1_0.pth?raw=true'
        elif width_mult == 0.8:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8.pth?raw=true'
        elif width_mult == 0.5:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_5.pth?raw=true'
        elif width_mult == 0.35:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35.pth?raw=true'
        elif width_mult == 0.25:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_25.pth?raw=true'
        elif width_mult == 0.1:
            weights_url = 'https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_1.pth?raw=true'
        else:
            raise ValueError('model with width_mult={} does not have pretrained weights'.format(width_mult))

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

    model = hbonet(pretrained=True, width_mult=1.0)
    summary(model, input_size=(3, 224, 224))

    input_tensor = torch.randn(1, 3, 224, 224)
    #import torch.onnx
    #torch.onnx.export(model, input_tensor, "hbonet_1.0.onnx", verbose=False)
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
