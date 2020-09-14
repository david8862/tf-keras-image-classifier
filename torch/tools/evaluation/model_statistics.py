#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate MACs & PARAMs of a PyTorch model.
"""
import os, sys, argparse
import torch
from thop import profile, profile_origin, clever_format

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
#from common.utils import get_custom_ops

def get_macs(model, input_tensor):
    #custom_ops_dict = get_custom_ops()
    custom_ops_dict = None

    macs, params = profile(model, inputs=(input_tensor, ), custom_ops=custom_ops_dict, verbose=True)
    macs, params = clever_format([macs, params], "%.3f")

    print('Total MACs: {}'.format(macs))
    print('Total PARAMs: {}'.format(params))


def main():
    parser = argparse.ArgumentParser(description='PyTorch model MACs & PARAMs checking tool')
    parser.add_argument('--model_path', type=str, required=True, help='model file to evaluate')
    parser.add_argument('--model_input_shape', type=str, required=True, help='model input image shape as <height>x<width>')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device)

    height, width = args.model_input_shape.split('x')
    input_tensor = torch.randn(1, 3, int(height), int(width)).to(device)

    get_macs(model, input_tensor)


if __name__ == '__main__':
    main()
