#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run inference for CNN classifier TensorRT model, only
fit for TensorRT 8.5 & later version. You need to install TensorRT
on NVIDIA GPU PC with CUDA/cuDNN.

Reference from:
    https://zhuanlan.zhihu.com/p/547624036
    https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/08-Advance/MultiOptimizationProfile/main.py
    https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/01-SimpleDemo/TensorRT8.5/main.py
    https://blog.csdn.net/u012863603/article/details/132345407
"""
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
import cv2
import tensorrt as trt
from cuda import cudart  # install with "pip install cuda-python"

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image
from common.utils import get_classes


def validate_classifier_model_tensorrt(engine, image_file, class_names, loop_count, output_path):
    # get I/O tensor info
    num_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]

    # get input tensor details
    input_tensor_names = [tensor_name for tensor_name in tensor_names if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT]
    input_tensor_shapes = [engine.get_tensor_shape(input_tensor_name) for input_tensor_name in input_tensor_names]
    num_input = len(input_tensor_names)

    # assume only 1 input tensor
    assert num_input == 1, 'invalid input tensor number.'

    # get model input shape
    batch, channel, height, width = input_tensor_shapes[0]
    model_input_shape = (height, width)

    # get output tensor details
    output_tensor_names = [tensor_name for tensor_name in tensor_names if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT]
    output_tensor_shapes = [engine.get_tensor_shape(output_tensor_name) for output_tensor_name in output_tensor_names]
    num_output = len(output_tensor_names)

    # assume only 1 output tensor
    assert num_output == 1, 'invalid output tensor number.'

    # check output class number
    num_classes = output_tensor_shapes[0][-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # create engine execution context
    context = engine.create_execution_context()
    context.set_optimization_profile_async(0, 0)  # use default stream

    # prepare memory buffer on host
    buffer_host = []
    for i in range(num_io):
        buffer_host.append(np.empty(context.get_tensor_shape(tensor_names[i]), dtype=trt.nptype(engine.get_tensor_dtype(tensor_names[i]))))

    # prepare memory buffer on GPU device
    buffer_device = []
    for i in range(num_io):
        buffer_device.append(cudart.cudaMalloc(buffer_host[i].nbytes)[1])

    # set address of all input & output data in device buffer
    for i in range(num_io):
        context.set_tensor_address(tensor_names[i], int(buffer_device[i]))

    # prepare input image
    image = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
    image_data = np.expand_dims(image_data, axis=0)

    # fill image data to host buffer
    buffer_host[0] = np.ascontiguousarray(image_data)

    # copy input data from host buffer to device buffer
    for i in range(num_input):
        cudart.cudaMemcpy(buffer_device[i], buffer_host[i].ctypes.data, buffer_host[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # do inference computation
    start = time.time()
    for i in range(loop_count):
        context.execute_async_v3(0)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # copy output data from device buffer into host buffer
    for i in range(num_input, num_io):
        cudart.cudaMemcpy(buffer_host[i].ctypes.data, buffer_device[i], buffer_host[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    prediction = buffer_host[1]
    handle_prediction(prediction, image_file, np.array(image), class_names, output_path)

    # free GPU memory buffer after all work
    for buffer in buffer_device:
        cudart.cudaFree(buffer)


def handle_prediction(prediction, image_file, image, class_names, output_path):
    indexes = np.argsort(prediction[0])
    indexes = indexes[::-1]
    #only pick top-1 class index
    index = indexes[0]
    score = prediction[0][index]

    result = '{name}:{conf:.3f}'.format(name=class_names[index] if class_names else index, conf=float(score))
    print('Class result\n', result)

    cv2.putText(image, result,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_file))
        Image.fromarray(image).save(output_file)
    else:
        Image.fromarray(image).show()
    return


def load_engine(model_path):
    # support TensorRT engine model
    if model_path.endswith('.engine'):
        # load model & create engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, mode='rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        raise ValueError('invalid model file')

    return engine


def main():
    parser = argparse.ArgumentParser(description='validate CNN classifier TensorRT model (.engine) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)

    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    #parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=False)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--output_path', help='output path to save predict result, default=%(default)s', type=str, required=False, default=None)

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        class_names = get_classes(args.classes_path)

    # param parse
    #height, width = args.model_input_shape.split('x')
    #model_input_shape = (int(height), int(width))

    # prepare environment
    assert trt.__version__ >= '8.5', 'invalid TensorRT version'
    cudart.cudaDeviceSynchronize()

    # load TensorRT engine
    engine = load_engine(args.model_path)

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    # loop the sample list to predict on each image
    for image_file in image_files:
        validate_classifier_model_tensorrt(engine, image_file, class_names, args.loop_count, args.output_path)


if __name__ == '__main__':
    main()
