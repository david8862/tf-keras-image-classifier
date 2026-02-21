#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton http/grpc client for classifier model

Reference from:
    https://blog.csdn.net/sgyuanshi/article/details/123536579
    https://github.com/QunBB/DeepLearning/blob/main/triton/client/_http.py
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
from common.data_utils import preprocess_image
from common.utils import get_classes


def classifier_http_client(server_addr, server_port, model_name, image_files, class_names, output_path):
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
    assert input_type == 'FP32', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(inputs_metadata[0]['shape']) == 4, 'invalid input shape.'

    # check if input layout is NHWC or NCHW
    if inputs_metadata[0]['shape'][1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = inputs_metadata[0]['shape']  #NCHW
    else:
        print("NHWC input layout")
        batch, height, width, channel = inputs_metadata[0]['shape']  #NHWC

    model_input_shape = (height, width)

    assert len(outputs_metadata[0]['shape']) == 2, 'invalid output shape.' # (1, num_classes)
    num_classes = outputs_metadata[0]['shape'][1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # loop the sample list to predict on each image
    for image_file in image_files:
        # prepare input image
        image = Image.open(image_file).convert('RGB')
        image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(httpclient.InferInput(input_name, image_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(image_data, binary_data=False)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=False))
        #outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=False, class_count=3))  # class_count for topN result

        # do inference to get prediction
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs, request_id=str(1), sequence_id=0, sequence_start=False, sequence_end=False, priority=0, timeout=None)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result, image_file, np.array(image), class_names, output_path)

    # close triton http client
    triton_client.close()



def grpc_data_type_str(data_type):
    if data_type == 0:
        return 'TYPE_INVALID'
    elif data_type == 1:
        return 'TYPE_BOOL'
    elif data_type == 2:
        return 'TYPE_UINT8'
    elif data_type == 3:
        return 'TYPE_UINT16'
    elif data_type == 4:
        return 'TYPE_UINT32'
    elif data_type == 5:
        return 'TYPE_UINT64'
    elif data_type == 6:
        return 'TYPE_INT8'
    elif data_type == 7:
        return 'TYPE_INT16'
    elif data_type == 8:
        return 'TYPE_INT32'
    elif data_type == 9:
        return 'TYPE_INT64'
    elif data_type == 10:
        return 'TYPE_FP16'
    elif data_type == 11:
        return 'TYPE_FP32'
    elif data_type == 12:
        return 'TYPE_FP64'
    elif data_type == 13:
        return 'TYPE_STRING'
    elif data_type == 14:
        return 'TYPE_BF16'
    else:
        raise ValueError('invalid data_type: ' + data_type)


def classifier_grpc_client(server_addr, server_port, model_name, image_files, class_names, output_path):
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
    assert input_type == 'FP32', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(inputs_metadata[0].shape) == 4, 'invalid input shape.'

    # check if input layout is NHWC or NCHW
    if inputs_metadata[0].shape[1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = inputs_metadata[0].shape  #NCHW
    else:
        print("NHWC input layout")
        batch, height, width, channel = inputs_metadata[0].shape  #NHWC

    model_input_shape = (height, width)

    assert len(outputs_metadata[0].shape) == 2, 'invalid output shape.' # (1, num_classes)
    num_classes = outputs_metadata[0].shape[1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # loop the sample list to predict on each image
    for image_file in image_files:
        # prepare input image
        image = Image.open(image_file).convert('RGB')
        image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(grpcclient.InferInput(input_name, image_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(image_data)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        #outputs.append(grpcclient.InferRequestedOutput(output_name, class_count=3))  # class_count for topN result

        # do inference to get prediction
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs, request_id=str(1), sequence_id=0, sequence_start=False, sequence_end=False, priority=0, timeout=None)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result, image_file, np.array(image), class_names, output_path)

    # close triton grpc client
    triton_client.close()



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



def main():
    parser = argparse.ArgumentParser(description='classifier http/grpc client for triton inference server')
    parser.add_argument('--server_addr', type=str, required=False, default='localhost',
        help='triton server address, default=%(default)s')
    parser.add_argument('--server_port', type=str, required=False, default='8000',
        help='triton server port (8000 for http & 8001 for grpc), default=%(default)s')
    parser.add_argument('--model_name', type=str, required=False, default='classifier_onnx',
        help='model name for inference, default=%(default)s')
    parser.add_argument('--image_path', type=str, required=True,
        help="image file or directory to inference")
    parser.add_argument('--classes_path', type=str, required=False, default=None,
        help='path to class name definitions')
    parser.add_argument('--output_path', type=str, required=False, default=None,
        help='output path to save inference result, default=%(default)s')
    parser.add_argument('--protocol', type=str, required=False, default='http', choices=['http', 'grpc'],
        help="comm protocol between triton server & client (http/grpc), default=%(default)s")

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        class_names = get_classes(args.classes_path)

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    if args.protocol == 'http':
        classifier_http_client(args.server_addr, args.server_port, args.model_name, image_files, class_names, args.output_path)
    elif args.protocol == 'grpc':
        classifier_grpc_client(args.server_addr, args.server_port, args.model_name, image_files, class_names, args.output_path)
    else:
        raise ValueError('invalid protocol: ' + args.protocol)


if __name__ == "__main__":
    main()
