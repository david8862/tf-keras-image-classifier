#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Reference from:
    https://blog.csdn.net/lai_cheng/article/details/118961420
    https://github.com/Laicheng0830/Pytorch_Model_Quantization
"""
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
import torch

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image


def evaluate(model, device, data_path, model_input_shape):
    image_files = glob.glob(os.path.join(data_path, '*'))

    for image_file in image_files:
        # prepare input image
        image = Image.open(image_file).convert('RGB')
        image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=True)
        image_data = image_data.unsqueeze(0).to(device)

        prediction = model(image_data)

    return



def torch_quant_convert(model_file, device, data_path, model_input_shape, output_file):
    # load fp32 model
    model_fp32 = torch.load(model_file, map_location=device).float()
    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    evaluate(model_fp32_prepared, device, data_path, model_input_shape)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)

    print("model int8", model_int8)

    # save model
    torch.save(model_int8, output_file)

    # export to onnx model
    # TODO: not work yet, need more debugging
    #img = torch.zeros((1, 3, *model_input_shape))
    #torch.onnx.export(model_int8, img, 'test.onnx', verbose=False, opset_version=12, input_names=['image_input'],
                      #output_names=['scores'])



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='PyTorch post training integer quantization converter')
    parser.add_argument('--model_file', type=str, required=True, help='input model file to quantize')
    parser.add_argument('--data_path', type=str, required=True, help='calibration image data path')
    parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--output_file', type=str, required=True, help='output quantized model file')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch_quant_convert(args.model_file, device, args.data_path, model_input_shape, args.output_file)


if __name__ == "__main__":
    main()
