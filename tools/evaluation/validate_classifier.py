#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import numpy as np
import cv2
from PIL import Image
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_custom_objects
from common.preprocess_crop import load_and_crop_img


def validate_classifier_model(model_path, image_file, class_names, model_image_size, loop_count):
    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
    K.set_learning_phase(0)

    num_classes = model.output.shape.as_list()[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input image
    ori_img = Image.open(image_file)
    img = load_and_crop_img(image_file, target_size=model_image_size, interpolation='nearest:center')
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

    handle_prediction(prediction, np.array(ori_img), class_names)



def validate_classifier_model_onnx(model_path, image_file, class_names, loop_count):
    sess = onnxruntime.InferenceSession(model_path)

    input_tensors = []
    for i, input_tensor in enumerate(sess.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    batch, height, width, channel = input_tensors[0].shape
    model_image_size = (height, width)

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
    ori_img = Image.open(image_file)
    img = load_and_crop_img(image_file, target_size=model_image_size, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = sess.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = sess.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], np.array(ori_img), class_names)



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

    model_image_size = (height, width)


    # prepare input image
    ori_img = Image.open(image_file)
    img = load_and_crop_img(image_file, target_size=model_image_size, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
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
    # our postprocess code based on TF channel last format, so if the output format
    # doesn't match, we need to transpose
    if output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        output_data = output_data.transpose((0,2,3,1))
    elif output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe_C4:
        raise ValueError('unsupported output tensor dimension type')

    prediction.append(output_data)
    handle_prediction(prediction[0], np.array(ori_img), class_names)



def validate_classifier_model_pb(model_path, image_file, class_names, loop_count):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we hardcode the input/output tensor names here to get them from model
    input_tensor_name = 'graph/image_input:0'
    output_tensor_name = 'graph/dense/Softmax:0'


    #load frozen pb graph
    def load_pb_graph(model_path):
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

    graph = load_pb_graph(model_path)

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in graph.get_operations():
        #print(op.name, op.values())

    image_input = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    batch, height, width, channel = image_input.shape
    model_image_size = (int(height), int(width))

    num_classes = output_tensor.shape[-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'


    # prepare input image
    ori_img = Image.open(image_file)
    img = load_and_crop_img(image_file, target_size=model_image_size, interpolation='nearest:center')
    image_data = np.array(img, dtype=np.float32) / 255.
    image_data = np.expand_dims(image_data, axis=0)


    # predict once first to bypass the model building time
    with tf.Session(graph=graph) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            image_input: image_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=graph) as sess:
                prediction = sess.run(output_tensor, feed_dict={
                    image_input: image_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction, np.array(ori_img), class_names)



def validate_classifier_model_tflite(model_path, image_file, class_names, loop_count):
    interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    model_image_size = (height, width)

    num_classes = output_details[0]['shape'][-1]
    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input image
    ori_img = Image.open(image_file)
    img = load_and_crop_img(image_file, target_size=model_image_size, interpolation='nearest:center')
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

    handle_prediction(prediction[0], np.array(ori_img), class_names)
    return


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
    parser = argparse.ArgumentParser(description='validate CNN classifier model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)

    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--model_image_size', help='model image input size as <height>x<width>, default=%(default)s', type=str, default='224x224')
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=False)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        class_names = get_classes(args.classes_path)

    # param parse
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))

    # support of tflite model
    if args.model_path.endswith('.tflite'):
        validate_classifier_model_tflite(args.model_path, args.image_file, class_names, args.loop_count)
    # support of MNN model
    elif args.model_path.endswith('.mnn'):
        validate_classifier_model_mnn(args.model_path, args.image_file, class_names, args.loop_count)
    # support of TF 1.x frozen pb model
    elif args.model_path.endswith('.pb'):
        validate_classifier_model_pb(args.model_path, args.image_file, class_names, args.loop_count)
    # support of ONNX model
    elif args.model_path.endswith('.onnx'):
        validate_classifier_model_onnx(args.model_path, args.image_file, class_names, args.loop_count)
    # normal keras h5 model
    elif args.model_path.endswith('.h5'):
        validate_classifier_model(args.model_path, args.image_file, class_names, model_image_size, args.loop_count)
    else:
        raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
