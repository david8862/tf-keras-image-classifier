#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import numpy as np
import cv2
from PIL import Image
import torch
import MNN
import onnxruntime
from operator import mul
from functools import reduce

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image
from common.utils import get_classes


def validate_classifier_model_torch(model_path, image_file, class_names, model_input_shape, loop_count):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # load model
    model = torch.load(model_path, map_location=device)
    model.eval()

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

    handle_prediction(prediction, np.array(image), class_names)



def validate_classifier_model_onnx(model_path, image_file, class_names, loop_count):
    sess = onnxruntime.InferenceSession(model_path)

    input_tensors = []
    for i, input_tensor in enumerate(sess.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    batch, channel, height, width = input_tensors[0].shape
    model_input_shape = (height, width)

    output_tensors = []
    for i, output_tensor in enumerate(sess.get_outputs()):
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
    prediction = sess.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = sess.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], np.array(image), class_names)



def validate_classifier_model_mnn(model_path, image_file, class_names, loop_count):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()

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


    # use a temp tensor to copy data
    tmp_input = MNN.Tensor(input_shape, input_tensor.getDataType(),\
                    image_data, input_tensor.getDimensionType())

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
    handle_prediction(prediction[0], np.array(image), class_names)



def handle_prediction(prediction, image, class_names):
    indexes = np.argsort(prediction[0])
    indexes = indexes[::-1]
    #only pick top-1 class index
    index = indexes[0]
    score = prediction[0][index]

    cv2.putText(image, '{name}:{conf:.3f}'.format(name=class_names[index] if class_names else index, conf=float(score)),
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255,0,0),
                thickness=1,
                lineType=cv2.LINE_AA)
    Image.fromarray(image).show()



def main():
    parser = argparse.ArgumentParser(description='validate CNN classifier model (pth/onnx/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)

    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=False)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        class_names = get_classes(args.classes_path)

    # param parse
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    # support of PyTorch pth model
    if args.model_path.endswith('.pth'):
        validate_classifier_model_torch(args.model_path, args.image_file, class_names, model_input_shape, args.loop_count)
    # support of ONNX model
    elif args.model_path.endswith('.onnx'):
        validate_classifier_model_onnx(args.model_path, args.image_file, class_names, args.loop_count)
    # support of MNN model
    elif args.model_path.endswith('.mnn'):
        validate_classifier_model_mnn(args.model_path, args.image_file, class_names, args.loop_count)
    else:
        raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
