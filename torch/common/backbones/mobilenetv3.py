#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


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


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    if 'pretrained' in kwargs.keys():
        pretrained = kwargs['pretrained']
        kwargs.pop('pretrained')
    else:
        pretrained = None

    if 'width_mult' in kwargs.keys():
        width_mult = kwargs['width_mult']
    else:
        width_mult = 1.

    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    model = MobileNetV3(cfgs, mode='large', **kwargs)

    if pretrained:
        # load pretrained weights
        if width_mult == 1.:
            weights_url = 'https://github.com/d-li14/mobilenetv3.pytorch/blob/210ebaefc563fa8dddb7b7a807f8b02ab9f31dff/pretrained/mobilenetv3-large-1cd25616.pth?raw=true'
        elif width_mult == 0.75:
            weights_url = 'https://github.com/d-li14/mobilenetv3.pytorch/blob/210ebaefc563fa8dddb7b7a807f8b02ab9f31dff/pretrained/mobilenetv3-large-0.75-9632d2a8.pth?raw=true'
        else:
            raise ValueError('model with width_mult={} does not have pretrained weights'.format(width_mult))

        pretrain_dict = model_zoo.load_url(weights_url)
        model.load_state_dict(pretrain_dict)

    return model


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
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

    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    model = MobileNetV3(cfgs, mode='small', **kwargs)

    if pretrained:
        # load pretrained weights
        if width_mult == 1.:
            weights_url = 'https://github.com/d-li14/mobilenetv3.pytorch/blob/210ebaefc563fa8dddb7b7a807f8b02ab9f31dff/pretrained/mobilenetv3-small-55df8e1f.pth?raw=true'
        elif width_mult == 0.75:
            weights_url = 'https://github.com/d-li14/mobilenetv3.pytorch/blob/210ebaefc563fa8dddb7b7a807f8b02ab9f31dff/pretrained/mobilenetv3-small-0.75-86c972c3.pth?raw=true'
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

    model = mobilenetv3_large(pretrained=True, width_mult=1.0)
    summary(model, input_size=(3, 224, 224))

    input_tensor = torch.randn(1, 3, 224, 224)
    #import torch.onnx
    #torch.onnx.export(model, input_tensor, "mobilenetv3_large_1.0.onnx", verbose=False)
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
