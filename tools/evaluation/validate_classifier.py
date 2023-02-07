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
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_custom_objects, optimize_tf_gpu
from common.preprocess_crop import load_and_crop_img


optimize_tf_gpu(tf, K)


def validate_classifier_model(model, image_file, class_names, model_input_shape, loop_count, output_path):
    num_classes = model.output.shape.as_list()[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input image
    ori_img = Image.open(image_file).convert('RGB')
    img = load_and_crop_img(image_file, target_size=model_input_shape, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)

    # predict once first to bypass the model building time
    model.predict([image_data])

    # get predict output
    start = time.time()
    for i in range(loop_count):
        prediction = model.predict([image_data])
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction, image_file, np.array(ori_img), class_names, output_path)


def validate_classifier_model_onnx(model, image_file, class_names, loop_count, output_path):
    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    # check if input layout is NHWC or NCHW
    if input_tensors[0].shape[1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = input_tensors[0].shape  #NCHW
    else:
        print("NHWC input layout")
        batch, height, width, channel = input_tensors[0].shape  #NHWC

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
    ori_img = Image.open(image_file).convert('RGB')
    img = load_and_crop_img(image_file, target_size=model_input_shape, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)

    if input_tensors[0].shape[1] == 3:
        # transpose image for NCHW layout
        image_data = image_data.transpose((0,3,1,2))

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = model.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = model.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], image_file, np.array(ori_img), class_names, output_path)



def validate_classifier_model_mnn(interpreter, session, image_file, class_names, loop_count, output_path):
    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)

    # get & resize input shape
    input_shape = list(input_tensor.getShape())
    if input_shape[0] == 0:
        input_shape[0] = 1
        interpreter.resizeTensor(input_tensor, tuple(input_shape))
        interpreter.resizeSession(session)

    # check if input layout is NHWC or NCHW
    if input_shape[1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = input_shape  #NCHW
    elif input_shape[-1] == 3:
        print("NHWC input layout")
        batch, height, width, channel = input_shape  #NHWC
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    model_input_shape = (height, width)

    # prepare input image
    ori_img = Image.open(image_file).convert('RGB')
    img = load_and_crop_img(image_file, target_size=model_input_shape, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)


    # create a temp tensor to copy data,
    # use TF NHWC layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, height, width, channel)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(image_data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

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
    handle_prediction(prediction[0], image_file, np.array(ori_img), class_names, output_path)



def validate_classifier_model_pb(model, image_file, class_names, loop_count, output_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we hardcode the input/output tensor names here to get them from model
    input_tensor_name = 'graph/image_input:0'
    output_tensor_name = 'graph/score_predict/Softmax:0'

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in model.get_operations():
        #print(op.name, op.values())

    image_input = model.get_tensor_by_name(input_tensor_name)
    output_tensor = model.get_tensor_by_name(output_tensor_name)

    batch, height, width, channel = image_input.shape
    model_input_shape = (int(height), int(width))

    num_classes = output_tensor.shape[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'


    # prepare input image
    ori_img = Image.open(image_file).convert('RGB')
    img = load_and_crop_img(image_file, target_size=model_input_shape, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)


    # predict once first to bypass the model building time
    with tf.Session(graph=model) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            image_input: image_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=model) as sess:
                prediction = sess.run(output_tensor, feed_dict={
                    image_input: image_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction, image_file, np.array(ori_img), class_names, output_path)



def validate_classifier_model_tflite(interpreter, image_file, class_names, loop_count, output_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    model_input_shape = (height, width)

    num_classes = output_details[0]['shape'][-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input image
    ori_img = Image.open(image_file).convert('RGB')
    img = load_and_crop_img(image_file, target_size=model_input_shape, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    handle_prediction(prediction[0], image_file, np.array(ori_img), class_names, output_path)
    return


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


#load TF 1.x frozen pb graph
def load_graph(model_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # We parse the graph_def file
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="graph",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_val_model(model_path):
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        custom_object_dict = get_custom_objects()

        model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model


def main():
    parser = argparse.ArgumentParser(description='validate CNN classifier model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)

    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    parser.add_argument('--model_input_shape', help='model image input shape as <height>x<width>, default=%(default)s', type=str, default='224x224')
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

    model = load_val_model(args.model_path)
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
        # support of tflite model
        if args.model_path.endswith('.tflite'):
            validate_classifier_model_tflite(model, image_file, class_names, args.loop_count, args.output_path)
        # support of MNN model
        elif args.model_path.endswith('.mnn'):
            validate_classifier_model_mnn(model, session, image_file, class_names, args.loop_count, args.output_path)
        # support of TF 1.x frozen pb model
        elif args.model_path.endswith('.pb'):
            validate_classifier_model_pb(model, image_file, class_names, args.loop_count, args.output_path)
        # support of ONNX model
        elif args.model_path.endswith('.onnx'):
            validate_classifier_model_onnx(model, image_file, class_names, args.loop_count, args.output_path)
        # normal keras h5 model
        elif args.model_path.endswith('.h5'):
            validate_classifier_model(model, image_file, class_names, model_input_shape, args.loop_count, args.output_path)
        else:
            raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
