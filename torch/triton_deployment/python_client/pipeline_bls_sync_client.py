#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton http/grpc client for classifier pipeline with synchronous bls wrapper
"""
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
import cv2
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes


def pipeline_bls_sync_http_client(server_addr, server_port, model_name, target_model_name, target_input_name, target_output_name, image_files, class_names, output_path):
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

    assert len(inputs_metadata) == 4, 'invalid input number.'
    assert len(outputs_metadata) == 1, 'invalid output number.'

    # go through inputs metadata list to get real input tensor name
    for i in range(len(inputs_metadata)):
        if inputs_metadata[i]["name"] == "input":
            input_name = inputs_metadata[i]['name']
            input_type = inputs_metadata[i]['datatype']

    output_name = outputs_metadata[0]['name']
    output_type = outputs_metadata[0]['datatype']

    # check input & output metadata
    assert input_type == 'UINT8', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(outputs_metadata[0]['shape']) == 2, 'invalid output shape.' # (1, num_classes)
    num_classes = outputs_metadata[0]['shape'][-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # loop the sample list to predict on each image
    for image_file in image_files:
        # load input image
        image_data = np.fromfile(image_file, dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(httpclient.InferInput("model_name", [1], np_to_triton_dtype(np.object_)))
        inputs[0].set_data_from_numpy(np.array([target_model_name], dtype=np.object_))

        inputs.append(httpclient.InferInput("model_input_name", [1], np_to_triton_dtype(np.object_)))
        inputs[1].set_data_from_numpy(np.array([target_input_name], dtype=np.object_))

        inputs.append(httpclient.InferInput("model_output_name", [1], np_to_triton_dtype(np.object_)))
        inputs[2].set_data_from_numpy(np.array([target_output_name], dtype=np.object_))

        inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
        inputs[3].set_data_from_numpy(image_data, binary_data=False)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=False))

        # do inference to get prediction
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs, request_id=str(1), sequence_id=0, sequence_start=False, sequence_end=False, priority=0, timeout=None)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result, image_file, class_names, output_path)

    # close triton http client
    triton_client.close()



def pipeline_bls_sync_grpc_client(server_addr, server_port, model_name, target_model_name, target_input_name, target_output_name, image_files, class_names, output_path):
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

    assert len(inputs_metadata) == 4, 'invalid input number.'
    assert len(outputs_metadata) == 1, 'invalid output number.'

    # go through inputs metadata list to get real input tensor name
    for i in range(len(inputs_metadata)):
        if inputs_metadata[i].name == "input":
            input_name = inputs_metadata[i].name
            input_type = inputs_metadata[i].datatype

    output_name = outputs_metadata[0].name
    output_type = outputs_metadata[0].datatype

    # check input & output metadata
    assert input_type == 'UINT8', 'invalid input type.'
    assert output_type == 'FP32', 'invalid output type.'

    assert len(outputs_metadata[0].shape) == 2, 'invalid output shape.' # (1, num_classes)
    num_classes = outputs_metadata[0].shape[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # loop the sample list to predict on each image
    for image_file in image_files:
        # load input image
        image_data = np.fromfile(image_file, dtype="uint8")
        image_data = np.expand_dims(image_data, axis=0)

        # prepare input/output list
        inputs = []
        inputs.append(grpcclient.InferInput("model_name", [1], np_to_triton_dtype(np.object_)))
        inputs[0].set_data_from_numpy(np.array([target_model_name], dtype=np.object_))

        inputs.append(grpcclient.InferInput("model_input_name", [1], np_to_triton_dtype(np.object_)))
        inputs[1].set_data_from_numpy(np.array([target_input_name], dtype=np.object_))

        inputs.append(grpcclient.InferInput("model_output_name", [1], np_to_triton_dtype(np.object_)))
        inputs[2].set_data_from_numpy(np.array([target_output_name], dtype=np.object_))

        inputs.append(grpcclient.InferInput(input_name, image_data.shape, "UINT8"))
        inputs[3].set_data_from_numpy(image_data)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))

        # do inference to get prediction
        start = time.time()
        prediction = triton_client.infer(model_name, inputs=inputs, outputs=outputs, request_id=str(1), sequence_id=0, sequence_start=False, sequence_end=False, priority=0, timeout=None)
        end = time.time()
        print("Inference time: {:.8f}ms".format((end - start) * 1000))

        result = prediction.as_numpy(output_name)
        handle_prediction(result, image_file, class_names, output_path)

    # close triton grpc client
    triton_client.close()



def handle_prediction(prediction, image_file, class_names, output_path):
    indexes = np.argsort(prediction[0])
    indexes = indexes[::-1]
    #only pick top-1 class index
    index = indexes[0]
    score = prediction[0][index]

    result = '{name}:{conf:.3f}'.format(name=class_names[index] if class_names else index, conf=float(score))
    print('Class result\n', result)

    image = np.array(Image.open(image_file).convert('RGB'))
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
    parser = argparse.ArgumentParser(description='classifier pipeline with synchronous bls wrapper classifier pipeline http/grpc client for triton inference server')
    parser.add_argument('--server_addr', type=str, required=False, default='localhost',
        help='triton server address, default=%(default)s')
    parser.add_argument('--server_port', type=str, required=False, default='8000',
        help='triton server port (8000 for http & 8001 for grpc), default=%(default)s')
    parser.add_argument('--model_name', type=str, required=False, default='classifier_pipeline_bls_sync',
        help='model name for inference, default=%(default)s')
    parser.add_argument('--target_model_name', type=str, required=False, default='classifier_pipeline',
        help='target model name for invoke, default=%(default)s')
    parser.add_argument('--target_input_name', type=str, required=False, default='input',
        help='target model input tensor name, default=%(default)s')
    parser.add_argument('--target_output_name', type=str, required=False, default='output',
        help='target model output name, default=%(default)s')
    parser.add_argument('--image_path', type=str, required=True,
        help="image file or directory to inference")
    parser.add_argument('--classes_path', type=str, required=False, default=None,
        help='path to class name definitions')
    parser.add_argument('--output_path', type=str, required=False, default=None,
        help='output path to save dumped model, default=%(default)s')
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
        pipeline_bls_sync_http_client(args.server_addr, args.server_port, args.model_name, args.target_model_name, args.target_input_name, args.target_output_name, image_files, class_names, args.output_path)
    elif args.protocol == 'grpc':
        pipeline_bls_sync_grpc_client(args.server_addr, args.server_port, args.model_name, args.target_model_name, args.target_input_name, args.target_output_name, image_files, class_names, args.output_path)
    else:
        raise ValueError('invalid protocol: ' + args.protocol)


if __name__ == "__main__":
    main()
