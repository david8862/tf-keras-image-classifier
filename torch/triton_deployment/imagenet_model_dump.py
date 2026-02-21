#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
#import torchinfo

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
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
    def __init__(self, model_type):
        super(Classifier, self).__init__()
        self.base_model = self.get_basemodel(model_type)

    def get_basemodel(self, model_type):
        if model_type == 'mobilenetv2':
            width_mult=0.5
            base_model = mobilenetv2(pretrained=True, width_mult=width_mult)
        elif model_type == 'mobilenetv3large':
            width_mult=0.75
            base_model = mobilenetv3_large(pretrained=True, width_mult=width_mult)
        elif model_type == 'mobilenetv3small':
            width_mult=0.75
            base_model = mobilenetv3_small(pretrained=True, width_mult=width_mult)
        elif model_type == 'peleenet':
            base_model = peleenet(pretrained=True, weights_path=None)
        elif model_type == 'csppeleenet':
            base_model = csppeleenet(pretrained=True)
        elif model_type == 'shufflenetv2':
            base_model = shufflenet_v2_x1_0(pretrained=True, progress=True)
        elif model_type == 'ghostnet':
            base_model = ghostnet(pretrained=True, weights_path=None)
        elif model_type == 'hbonet':
            width_mult=0.5
            base_model = hbonet(pretrained=True, width_mult=width_mult)
        elif model_type == 'mobilevit_s':
            base_model = mobilevit_s(pretrained=False)
        elif model_type == 'mobilevit_xs':
            base_model = mobilevit_xs(pretrained=False)
        elif model_type == 'mobilevit_xxs':
            base_model = mobilevit_xxs(pretrained=False)
        elif model_type == 'resnet50':
            base_model = resnet50(pretrained=True, progress=True)
        else:
            raise ValueError('Unsupported model type')

        return base_model


    def forward(self, x):
        x = self.base_model(x)
        x = F.softmax(x, dim=1)
        return x



def model_dump(model_type, model_input_shape, batch_size, output_path):
    device = torch.device("cpu")

    # get model
    model = Classifier(model_type).to(device)
    torchsummary.summary(model, input_size=(3,)+model_input_shape, batch_size=1, device='cpu')
    #torchinfo.summary(model, input_size=(1, 3)+model_input_shape, device=device)

    os.makedirs(output_path, exist_ok=True)

    if batch_size == -1:
        input_tensor = torch.randn((1, 3, *model_input_shape), device=device)

        # dump dynamic batch-size onnx model
        torch.onnx.export(model, input_tensor, os.path.join(output_path, 'model.onnx'), verbose=False, opset_version=12, input_names=['image_input'], output_names=['scores'],
                          dynamic_axes={"image_input": {0: "batch_size"}, "scores": {0: "batch_size"}})
    else:
        input_tensor = torch.randn((batch_size, 3, *model_input_shape), device=device)

        # dump fix batch-size onnx model
        torch.onnx.export(model, input_tensor, os.path.join(output_path, 'model.onnx'), verbose=False, opset_version=12, input_names=['image_input'], output_names=['scores'])


    # dump torchscript model
    torchscript_model = torch.jit.trace(model, input_tensor)
    torchscript_model.save(os.path.join(output_path, 'model.pt'))

    return



def main():
    parser = argparse.ArgumentParser(description='Dump imagenet pretrained CNN classifier model and convert to onnx/torchscript')
    parser.add_argument('--model_type', type=str, required=False, default='csppeleenet',
        help='model type: mobilenetv3/v2/csppeleenet, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='224x224',
        help="model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--batch_size', type=int, required=False, default=-1,
        help="batch size for inference, default=%(default)s")
    parser.add_argument('--output_path', type=str, required=False, default=os.path.join('model_repository', 'classifier_onnx', '1'),
        help='output path to save dumped model, default=%(default)s')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    model_dump(args.model_type, args.model_input_shape, args.batch_size, args.output_path)
    print('\nDone. Dumped model has been saved to', args.output_path)


if __name__ == "__main__":
    main()
