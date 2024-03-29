#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['mobilenetv2']


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


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
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


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
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

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
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

    model = MobileNetV2(**kwargs)

    if pretrained:
        # load pretrained weights
        if width_mult == 1.:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth?raw=true'
        elif width_mult == 0.75:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.75-dace9791.pth?raw=true'
        elif width_mult == 0.5:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth?raw=true'
        elif width_mult == 0.35:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.35-b2e15951.pth?raw=true'
        elif width_mult == 0.25:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.25-b61d2159.pth?raw=true'
        elif width_mult == 0.1:
            weights_url = 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.1-7d1d638a.pth?raw=true'
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

    model = mobilenetv2(pretrained=True, width_mult=1.0)
    summary(model, input_size=(3, 224, 224))

    input_tensor = torch.randn(1, 3, 224, 224)
    #import torch.onnx
    #torch.onnx.export(model, input_tensor, "mobilenetv2_1.0.onnx", verbose=False)
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
