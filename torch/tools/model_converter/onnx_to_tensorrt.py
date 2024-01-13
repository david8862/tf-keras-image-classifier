#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert CNN classifier ONNX model to TensorRT model

You need to install TensorRT on NVIDIA GPU PC with CUDA/cuDNN. See
    https://zhuanlan.zhihu.com/p/547624036

Reference from:
    https://blog.csdn.net/weixin_39263657/article/details/130687150
    https://zhuanlan.zhihu.com/p/547624036
    https://zhuanlan.zhihu.com/p/547969908
"""
import os, sys, argparse
import onnx
import tensorrt as trt


def onnx_to_tensorrt(input_model, output_model, fp16):
    onnx_model = onnx.load(input_model)

    input_tensors = onnx_model.graph.input

    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    input_shape = input_tensors[0].type.tensor_type.shape.dim

    # get input tensor shape
    input_batch = input_shape[0].dim_value
    input_channel = input_shape[1].dim_value
    input_height = input_shape[2].dim_value
    input_width = input_shape[3].dim_value

    if input_batch == 0:
        input_batch = 1

    # NCHW layout
    input_shape_list = [input_batch, input_channel, input_height, input_width]
    print("input shape:", input_shape_list)

    # create builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # create config for optimize model with trt
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<20) # 1MB

    if fp16:
        # set precision flag. will need calibration if use INT8
        config.set_flag(trt.BuilderFlag.FP16)

    # create network with dynamic batch: EXPLICIT_BATCH
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # create onnx parser
    parser = trt.OnnxParser(network, logger)


    ###########################################################################################
    # routine 1, parse onnx from object and apply optimization profile, need to know input shape
    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    profile = builder.create_optimization_profile()

    profile.set_shape('input', input_shape_list, input_shape_list, input_shape_list)
    config.add_optimization_profile(profile)


    ###########################################################################################
    # routine 2, parse onnx directly from model file

    #success = parser.parse_from_file(input_model)
    ## handle error
    #for idx in range(parser.num_errors):
        #print(parser.get_error(idx))
    #if not success:
        #pass  # Error handling code here




    # create serialized engine
    serialized_engine = builder.build_serialized_network(network, config)

    with open(output_model, mode='wb') as f:
        f.write(serialized_engine)
        print("generating tensorrt model file done!")



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert CNN classifier ONNX model to TensorRT model')
    parser.add_argument('--input_model', required=True, type=str, help='input ONNX model file')
    parser.add_argument('--output_model', required=True, type=str, help='output TensorRT .engine model file')
    parser.add_argument('--fp16', default=False, action="store_true", help='use fp16 precision for convert')

    args = parser.parse_args()

    onnx_to_tensorrt(args.input_model, args.output_model, args.fp16)


if __name__ == '__main__':
    main()

