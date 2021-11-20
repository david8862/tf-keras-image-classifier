#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
import cv2
from operator import mul
from functools import reduce
import torch
import MNN
import onnxruntime

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image
from common.utils import get_classes


def validate_classifier_model_torch(model, device, image_file, class_names, model_input_shape, loop_count, output_path):
    # prepare input image
    image = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=True)
    image_data = image_data.unsqueeze(0).to(device)

    with torch.no_grad():
        # predict once first to bypass the model building time
        prediction = model(image_data)

        num_classes = list(prediction.shape)[-1]
        if class_names:
            # check if classes number match with model prediction
            assert num_classes == len(class_names), 'classes number mismatch with model.'

        # get predict output
        start = time.time()
        for i in range(loop_count):
            prediction = model(image_data).cpu().numpy()
        end = time.time()
        print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction, image_file, np.array(image), class_names, output_path)


def validate_classifier_model_onnx(model, image_file, class_names, loop_count, output_path):
    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    batch, channel, height, width = input_tensors[0].shape
    model_input_shape = (height, width)

    output_tensors = []
    for i, output_tensor in enumerate(model.get_outputs()):
        output_tensors.append(output_tensor)
    # assume only 1 output tensor
    assert len(output_tensors) == 1, 'invalid output tensor number.'

    num_classes = output_tensors[0].shape[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input image
    image = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
    image_data = np.expand_dims(image_data, axis=0)

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = model.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = model.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], image_file, np.array(image), class_names, output_path)


def validate_classifier_model_mnn(interpreter, session, image_file, class_names, loop_count, output_path):
    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()
    if input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Tensorflow:
        batch, height, width, channel = input_shape
    elif input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        batch, channel, height, width = input_shape
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    model_input_shape = (height, width)

    # prepare input image
    image = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
    image_data = np.expand_dims(image_data, axis=0)


    # create a temp tensor to copy data,
    # use Caffe NCHW layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, channel, height, width)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(image_data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Caffe)

    # predict once first to bypass the model building time
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    start = time.time()
    for i in range(loop_count):
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    output_elementsize = reduce(mul, output_shape)

    num_classes = output_shape[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                #np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)

    prediction.append(output_data)
    handle_prediction(prediction[0], image_file, np.array(image), class_names, output_path)


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


def load_val_model(model_path, device):
    # support of PyTorch pth model
    if model_path.endswith('.pth'):
        model = torch.load(model_path, map_location=device)
        model.eval()

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)

    else:
        raise ValueError('invalid model file')

    return model


def main():
    parser = argparse.ArgumentParser(description='validate CNN classifier model (pth/onnx/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)

    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=False)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--output_path', help='output path to save predict result, default=%(default)s', type=str, required=False, default=None)

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        class_names = get_classes(args.classes_path)

    # param parse
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_val_model(args.model_path, device)
    if args.model_path.endswith('.mnn'):
        #MNN inference engine need create session
        session = model.createSession()

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    # loop the sample list to predict on each image
    for image_file in image_files:
        # support of PyTorch pth model
        if args.model_path.endswith('.pth'):
            validate_classifier_model_torch(model, device, image_file, class_names, model_input_shape, args.loop_count, args.output_path)
        # support of ONNX model
        elif args.model_path.endswith('.onnx'):
            validate_classifier_model_onnx(model, image_file, class_names, args.loop_count, args.output_path)
        # support of MNN model
        elif args.model_path.endswith('.mnn'):
            validate_classifier_model_mnn(model, session, image_file, class_names, args.loop_count, args.output_path)
        else:
            raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
