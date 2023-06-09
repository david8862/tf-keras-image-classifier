#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Reference from:
    https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    https://blog.csdn.net/m0_63642362/article/details/124741589
    https://www.cnblogs.com/ysnow/p/16773127.html
"""
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
import onnxruntime
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image



class DataReader(CalibrationDataReader):
    def __init__(self, data_path, model_file=None):
        self.data_path = data_path
        self.model_file = model_file
        self.preprocess_flag = True
        self.feed_dicts = []

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            model = onnxruntime.InferenceSession(self.model_file, None)

            input_tensors = []
            for i, input_tensor in enumerate(model.get_inputs()):
                input_tensors.append(input_tensor)
            # assume only 1 input tensor for image
            assert len(input_tensors) == 1, 'invalid input tensor number.'

            batch, channel, height, width = input_tensors[0].shape
            model_input_shape = (height, width)

            image_files = glob.glob(os.path.join(self.data_path, '*'))

            feed_list = []
            for image_file in image_files:
                # prepare input image
                image = Image.open(image_file).convert('RGB')
                image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
                image_data = np.expand_dims(image_data, axis=0)

                feed = {input_tensors[0].name: image_data}
                feed_list.append(feed)

            self.feed_dicts = iter(feed_list)
        return next(self.feed_dicts, None)



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='ONNX post training integer quantization converter')

    # common options
    parser.add_argument('--model_file', type=str, required=True,
        help='input onnx model file to quantize')
    parser.add_argument('--quantize_way', type=str, required=False, default='static', choices=['static', 'dynamic'],
        help = "way to quantize model (static/dynamic), default=%(default)s")
    parser.add_argument('--quantize_type', type=str, required=False, default='uint8', choices=['uint8', 'int8'],
        help = "model quantize data type (uint8/int8), default=%(default)s")

    # static quantization only options
    parser.add_argument('--quantize_format', type=str, required=False, default='QDQ', choices=['QDQ', 'QOperator'],
        help = "model quantize format (QDQ/QOperator), only need for static quantization, default=%(default)s")
    parser.add_argument('--calibrate_method', type=str, required=False, default='minmax', choices=['minmax', 'entropy', 'percentile'],
        help = "data calibration method (minmax/entropy/percentile), only need for static quantization, default=%(default)s")
    parser.add_argument('--data_path', type=str, required=False, default=None,
        help='calibration image data path. only need for static quantization')

    parser.add_argument('--output_file', type=str, required=True,
        help='output quantized model file')

    args = parser.parse_args()

    if args.quantize_type == 'uint8':
        quant_type = QuantType.QUInt8
    elif args.quantize_type == 'int8':
        quant_type = QuantType.QInt8
    else:
        raise ValueError('invalid quantize type')


    if args.quantize_way == 'static':

        if args.quantize_format == 'QDQ':
            quant_format = QuantFormat.QDQ
        elif args.quantize_format == 'QOperator':
            quant_format = QuantFormat.QOperator
        else:
            raise ValueError('invalid quantize format')

        if args.calibrate_method == 'minmax':
            calib_method = CalibrationMethod.MinMax
        elif args.calibrate_method == 'entropy':
            calib_method = CalibrationMethod.Entropy
        elif args.calibrate_method == 'percentile':
            calib_method = CalibrationMethod.Percentile
        else:
            raise ValueError('invalid calibrate method')

        # some extra options to control quantization
        extra_options = {'extra.Sigmoid.nnapi': False,
                         'ActivationSymmetric': False,
                         'WeightSymmetric': True,
                         'EnableSubgraph': False,
                         'ForceQuantizeNoInputCheck': False,
                         'MatMulConstBOnly': False,
                         'AddQDQPairToWeight': False,
                         'OpTypesToExcludeOutputQuantization': [],
                         'DedicatedQDQPair': False,
                         'QDQOpTypePerChannelSupportToAxis': {},
                         'CalibTensorRangeSymmetric': False,
                         'CalibMovingAverage': False,
                         'CalibMovingAverageConstant': 0.01,
                        }

        assert args.data_path is not None, 'need to provide calibration data path for static quantize'

        # data reader class to load & feed calibration input data
        data_reader = DataReader(args.data_path, args.model_file)
        # static quantize
        quantize_static(model_input=args.model_file,
                        model_output=args.output_file,
                        calibration_data_reader=data_reader,
                        quant_format=quant_format,
                        calibrate_method=calib_method,
                        weight_type=quant_type,
                        activation_type=quant_type,
                        per_channel=False,
                        optimize_model=False,
                        extra_options=extra_options)

    elif args.quantize_way == 'dynamic':
        # some extra options to control quantization
        extra_options = {'extra.Sigmoid.nnapi': False,
                         'ActivationSymmetric': False,
                         'WeightSymmetric': True,
                         'EnableSubgraph': False,
                         'ForceQuantizeNoInputCheck': False,
                         'MatMulConstBOnly': True,
                        }

        # dynamic quantize
        quantize_dynamic(model_input=args.model_file,
                         model_output=args.output_file,
                         weight_type=quant_type,
                         per_channel=False,
                         optimize_model=False,
                         extra_options=extra_options)
    else:
        raise ValueError('invalid quantize way')


if __name__ == "__main__":
    main()
