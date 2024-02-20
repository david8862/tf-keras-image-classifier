#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton http/grpc client for classifier preprocess model
"""
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
import cv2
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import denormalize_image


def classifier_preprocess_http_client(server_addr, server_port, model_name, image_files, output_path):
    # init triton http client
    server_url = server_addr + ':' + server_port
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False, ssl=False, ssl_options={}, insecure=False, ssl_context_factory=None)

    # check if triton server & target model is ready
    assert triton_client.is_server_live() & triton_client.is_server_ready(), 'Triton server is not ready'
    assert triton_client.is_model_ready(model_name), 'model ' + model_name + ' is not ready'

    # get input/output config & metadata, here
    # we use metadata to parse model info
    model_metadata = triton_client.get_model_metadata(model_name)
    model_config = triton_client.get_model_config(model_name)

    inputs_metadata = model_metadata['inputs']
    outputs_metadata = model_metadata['outputs']

    assert len(inputs_metadata) == 1, 'invalid input number.'
    assert len(outputs_metadata) == 1, 'invalid output number.'

    input_name = inputs_metadata[0]['name']
    input_type = inputs_metadata[0]['datatype']
    output_name = outputs_metadata[0]['name']
    output_type = outputs_metadata[0]['datatype']

    # check input & output metadata
    assert input_type == 'UINT8', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(outputs_metadata[0]['shape']) == 4, 'invalid output shape.'

    # check if output layout is NHWC or NCHW
    if outputs_metadata[0]['shape'][1] == 3:
        print("NCHW output layout")
        output_layout = "NCHW"
        batch, channel, height, width = outputs_metadata[0]['shape']  #NCHW
    else:
        print("NHWC output layout")
        output_layout = "NHWC"
        batch, height, width, channel = outputs_metadata[0]['shape']  #NHWC

    # loop the sample list to predict on each image
    for image_file in image_files:
        # load input image
        image_data = np.fromfile(image_file, dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
        inputs[0].set_data_from_numpy(image_data, binary_data=False)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=False))

        # do inference to get preprocess result
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result[0], output_layout, image_file, output_path)

    # close triton http client
    triton_client.close()



def classifier_preprocess_grpc_client(server_addr, server_port, model_name, image_files, output_path):
    # init triton grpc client
    server_url = server_addr + ':' + server_port
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False, ssl=False)

    # check if triton server & target model is ready
    assert triton_client.is_server_live() & triton_client.is_server_ready(), 'Triton server is not ready'
    assert triton_client.is_model_ready(model_name), 'model ' + model_name + ' is not ready'

    # get input/output config & metadata, here
    # we use metadata to parse model info
    model_metadata = triton_client.get_model_metadata(model_name)
    model_config = triton_client.get_model_config(model_name)

    inputs_metadata = model_metadata.inputs
    outputs_metadata = model_metadata.outputs

    assert len(inputs_metadata) == 1, 'invalid input number.'
    assert len(outputs_metadata) == 1, 'invalid output number.'

    input_name = inputs_metadata[0].name
    input_type = inputs_metadata[0].datatype
    output_name = outputs_metadata[0].name
    output_type = outputs_metadata[0].datatype

    # check input & output metadata
    assert input_type == 'UINT8', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(outputs_metadata[0].shape) == 4, 'invalid output shape.'

    # check if input layout is NHWC or NCHW
    if outputs_metadata[0].shape[1] == 3:
        print("NCHW output layout")
        output_layout = "NCHW"
        batch, channel, height, width = outputs_metadata[0].shape  #NCHW
    else:
        print("NHWC output layout")
        output_layout = "NHWC"
        batch, height, width, channel = outputs_metadata[0].shape  #NHWC

    # loop the sample list to predict on each image
    for image_file in image_files:
        # load input image
        image_data = np.fromfile(image_file, dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(grpcclient.InferInput(input_name, image_data.shape, "UINT8"))
        inputs[0].set_data_from_numpy(image_data)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))

        # do inference to get preprocess result
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result[0], output_layout, image_file, output_path)

    # close triton grpc client
    triton_client.close()



def handle_prediction(image_data, layout, image_file, output_path):
    # convert image data to channel last
    if layout == "NCHW":
        image_data = np.transpose(image_data, (1, 2, 0))

    # here we denormalize the preprocessed data back
    # to image, to check if the preprocess model
    # works correct
    image = denormalize_image(image_data)

    # save or show image
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_file))
        Image.fromarray(image).save(output_file)
    else:
        Image.fromarray(image).show()
    return



def main():
    parser = argparse.ArgumentParser(description='classifier preprocess http/grpc client for triton inference server')
    parser.add_argument('--server_addr', type=str, required=False, default='localhost',
        help='triton server address, default=%(default)s')
    parser.add_argument('--server_port', type=str, required=False, default='8000',
        help='triton server port (8000 for http & 8001 for grpc), default=%(default)s')
    parser.add_argument('--model_name', type=str, required=False, default='classifier_preprocess',
        help='model name for inference, default=%(default)s')
    parser.add_argument('--image_path', type=str, required=True,
        help="image file or directory to inference")
    parser.add_argument('--output_path', type=str, required=False, default=None,
        help='output path to save dumped model, default=%(default)s')
    parser.add_argument('--protocol', type=str, required=False, default='http', choices=['http', 'grpc'],
        help="comm protocol between triton server & client (http/grpc), default=%(default)s")

    args = parser.parse_args()

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    if args.protocol == 'http':
        classifier_preprocess_http_client(args.server_addr, args.server_port, args.model_name, image_files, args.output_path)
    elif args.protocol == 'grpc':
        classifier_preprocess_grpc_client(args.server_addr, args.server_port, args.model_name, image_files, args.output_path)
    else:
        raise ValueError('invalid protocol: ' + args.protocol)


if __name__ == "__main__":
    main()
