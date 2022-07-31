#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
#from torchvision.models import mobilenet_v2
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from common.backbones.mobilenetv2 import mobilenetv2
from common.backbones.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from common.backbones.peleenet import peleenet
from common.backbones.csppeleenet import csppeleenet
from common.backbones.ghostnet import ghostnet
from common.backbones.hbonet import hbonet
from common.backbones.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs


class Classifier(nn.Module):
    def __init__(self, model_type, num_classes, head_conv_channel):
        super(Classifier, self).__init__()
        self.head_conv_channel = head_conv_channel
        self.features, features_channel = self.get_features(model_type)

        self.head_conv = nn.Conv2d(in_channels=features_channel, out_channels=head_conv_channel, kernel_size=1)
        self.fc = nn.Linear(in_features=head_conv_channel, out_features=num_classes, bias=True)


    def get_features(self, model_type):
        if model_type == 'mobilenetv2':
            #model = mobilenet_v2(pretrained=True, progress=True)
            #features_channel = 1280
            width_mult=0.5
            model = mobilenetv2(pretrained=True, width_mult=width_mult)
            features_channel = int(320*width_mult)
            features = model.features
        elif model_type == 'mobilenetv3large':
            width_mult=0.75
            model = mobilenetv3_large(pretrained=True, width_mult=width_mult)
            features_channel = int(160*width_mult)
            features = model.features
        elif model_type == 'mobilenetv3small':
            width_mult=0.75
            model = mobilenetv3_small(pretrained=True, width_mult=width_mult)
            features_channel = int(96*width_mult)
            features = model.features
        elif model_type == 'peleenet':
            model = peleenet(pretrained=True, weights_path=None)
            features_channel = 704
            features = model.features
        elif model_type == 'csppeleenet':
            model = csppeleenet(pretrained=True)
            features_channel = 704
            features = nn.Sequential(model.stem,
                                     model.stages,
                                    )
        elif model_type == 'shufflenetv2':
            model = shufflenet_v2_x1_0(pretrained=True, progress=True)
            features_channel = 1024
            features = nn.Sequential(model.conv1,
                                     model.maxpool,
                                     model.stage2,
                                     model.stage3,
                                     model.stage4,
                                     model.conv5,
                                    )
        elif model_type == 'ghostnet':
            model = ghostnet(pretrained=True, weights_path=None)
            features_channel = 160
            features = model.features
        elif model_type == 'hbonet':
            width_mult=0.5
            model = hbonet(pretrained=True, width_mult=width_mult)
            features_channel = int(400*width_mult)
            features = model.features
        elif model_type == 'mobilevit_s':
            model = mobilevit_s(pretrained=False)
            features_channel = 160
            features = model.features
        elif model_type == 'mobilevit_xs':
            model = mobilevit_xs(pretrained=False)
            features_channel = 96
            features = model.features
        elif model_type == 'mobilevit_xxs':
            model = mobilevit_xxs(pretrained=False)
            features_channel = 80
            features = model.features
        elif model_type == 'resnet50':
            model = resnet50(pretrained=True, progress=True)
            features_channel = 2048
            features = nn.Sequential(model.conv1,
                                     model.bn1,
                                     model.relu,
                                     model.maxpool,
                                     model.layer1,
                                     model.layer2,
                                     model.layer3,
                                     model.layer4,
                                    )
        else:
            raise ValueError('Unsupported model type')
        return features, features_channel

    def classifier(self, x):
        x = self.head_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, self.head_conv_channel)
        x = self.fc(x)
        #x = F.log_softmax(x, dim=1)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

