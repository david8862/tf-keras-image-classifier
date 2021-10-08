#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, time
import numpy as np
from tqdm import tqdm

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
import MNN
import onnxruntime

from classifier.data import get_data_generator
from common.utils import get_classes, optimize_tf_gpu, get_custom_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


def predict_keras(model, data, target, class_index):
    output = model.predict(data)
    pred = np.argmax(output, axis=-1)
    target = np.argmax(target, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[:, class_index]
    return correct, class_score


def predict_pb(model, data, target, class_index):
    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we need to hardcode the input/output tensor names here to get them from model
    output_tensor_name = 'graph/dense/Softmax:0'

    # assume only 1 input tensor for image
    input_tensor_name = 'graph/image_input:0'

    # get input/output tensors
    image_input = model.get_tensor_by_name(input_tensor_name)
    output_tensor = model.get_tensor_by_name(output_tensor_name)

    with tf.Session(graph=model) as sess:
        output = sess.run(output_tensor, feed_dict={
            image_input: data
        })
    pred = np.argmax(output, axis=-1)
    target = np.argmax(target, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[:, class_index]
    return correct, class_score


def predict_tflite(interpreter, data, target, class_index):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    output = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        output.append(output_data)

    pred = np.argmax(output[0], axis=-1)
    target = np.argmax(target, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[0][:, class_index]
    return correct, class_score


def predict_onnx(model, data, target, class_index):

    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    feed = {input_tensors[0].name: data}
    output = model.run(None, feed)

    pred = np.argmax(output, axis=-1)
    target = np.argmax(target, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[0][:, class_index]
    return correct, class_score


def predict_mnn(interpreter, session, data, target, class_index):
    from functools import reduce
    from operator import mul

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

    # create a temp tensor to copy data,
    # use TF NHWC layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, height, width, channel)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    output = []
    # we only handle single output model
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    output_elementsize = reduce(mul, output_shape)
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    #tmp_output.printTensorData()

    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)

    output.append(output_data)
    pred = np.argmax(output[0], axis=-1)
    target = np.argmax(target, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[0][:, class_index]
    return correct, class_score


def threshold_search(class_scores, class_labels, class_index):
    '''
    walk through the score list to get a best threshold
    which can make highest accuracy
    '''
    class_scores = np.asarray(class_scores)
    class_labels = np.asarray(class_labels)
    best_accuracy = 0
    best_threshold = 0

    for i in range(len(class_scores)):
        # choose one score as threshold
        threshold = class_scores[i]
        # check predict label under this threshold
        y_pred = (class_scores >= threshold)
        accuracy = np.mean((y_pred == (class_labels == class_index).astype(int)).astype(int))

        # record best accuracy and threshold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return (best_accuracy, best_threshold)


def evaluate_accuracy(model, model_format, eval_generator, class_index):
    correct = 0.0
    class_scores = []
    class_labels = []

    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    step = eval_generator.samples // eval_generator.batch_size
    pbar = tqdm(total=step)
    for i in range(step):
        data, target = eval_generator.next()
        # normal keras h5 model
        if model_format == 'H5':
            tmp_correct, class_score = predict_keras(model, data, target, class_index)
            correct += tmp_correct
        # support of TF 1.x frozen pb model
        elif model_format == 'PB':
            tmp_correct, class_score = predict_pb(model, data, target, class_index)
            correct += tmp_correct
        # support of tflite model
        elif model_format == 'TFLITE':
            tmp_correct, class_score = predict_tflite(model, data, target, class_index)
            correct += tmp_correct
        # support of ONNX model
        elif model_format == 'ONNX':
            tmp_correct, class_score = predict_onnx(model, data, target, class_index)
            correct += tmp_correct
        # support of MNN model
        elif model_format == 'MNN':
            tmp_correct, class_score = predict_mnn(model, session, data, target, class_index)
            correct += tmp_correct
        else:
            raise ValueError('invalid model format')

        # record score & labels for specified class
        class_scores.append(float(class_score))
        class_labels.append(int(np.argmax(target, axis=-1)))

        pbar.set_description('Evaluate acc: %06.4f' % (correct/((i + 1)*(eval_generator.batch_size))))
        pbar.update(1)
    pbar.close()

    val_acc = correct / eval_generator.samples
    print('Test set accuracy: {}/{} ({:.2f}%)'.format(
        correct, eval_generator.samples, val_acc))

    # search for a best score threshold on one class
    accuracy, threshold = threshold_search(class_scores, class_labels, class_index)
    print('Best accuracy for class[{}]: {:.4f}, with score threshold {:.4f}'.format(class_index, accuracy, threshold))

    return val_acc


#load TF 1.x frozen pb graph
def load_graph(model_path):
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


def load_eval_model(model_path):
    # normal keras h5 model
    if model_path.endswith('.h5'):
        custom_object_dict = get_custom_objects()

        model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
        model_format = 'H5'
        K.set_learning_phase(0)

    # support of tflite model
    elif model_path.endswith('.tflite'):
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)
        model_format = 'PB'

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)
        model_format = 'ONNX'

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)
        model_format = 'MNN'

    else:
        raise ValueError('invalid model file')

    return model, model_format


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate CNN classifer model (h5/pb/onnx/tflite/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to evaluation image dataset')

    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to class definitions', default=None)

    parser.add_argument(
        '--model_input_shape', type=str,
        help='model image input size as <height>x<width>, default=%(default)s', default='224x224')

    parser.add_argument(
        '--class_index', type=int, required=False,
        help='class index to check the best threshold, default=%(default)s', default=0)

    args = parser.parse_args()

    # param parse
    if args.classes_path:
        class_names = get_classes(args.classes_path)
    else:
        class_names = None

    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    # get eval model
    model, model_format = load_eval_model(args.model_path)

    # eval data generator
    batch_size = 1
    eval_generator = get_data_generator(args.dataset_path, args.model_input_shape, batch_size, class_names, mode='eval')

    start = time.time()
    evaluate_accuracy(model, model_format, eval_generator, args.class_index)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
