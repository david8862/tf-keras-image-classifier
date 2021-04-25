#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import numpy as np
from tqdm import tqdm

import torch
import MNN
import onnxruntime

from classifier.data import get_dataloader


def predict_torch(model, device, data, target, class_index):
    model.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        correct = pred.eq(target.view_as(pred)).sum().item()
        class_score = output.detach().cpu().numpy()[:, class_index]
    return correct, class_score


def predict_onnx(model, data, target, class_index):
    # convert pytorch tensor to numpy array
    data, target = data.detach().cpu().numpy(), target.detach().cpu().numpy()

    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    feed = {input_tensors[0].name: data}
    output = model.run(None, feed)

    pred = np.argmax(output, axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output[0][:, class_index]
    return correct, class_score


def predict_mnn(interpreter, session, data, target, class_index):
    from functools import reduce
    from operator import mul

    # convert pytorch tensor to numpy array
    data, target = data.detach().cpu().numpy(), target.detach().cpu().numpy()

    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()

    # use a temp tensor to copy data
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    input_elementsize = reduce(mul, input_shape)
    tmp_input = MNN.Tensor(input_shape, input_tensor.getDataType(),\
                    tuple(data.reshape(input_elementsize, -1)), input_tensor.getDimensionType())

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    pred = []
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

    pred.append(output_data)
    pred = np.argmax(pred[0], axis=-1)
    correct = float(np.equal(pred, target).astype(np.int).sum())

    class_score = output_data[:, class_index]
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


def evaluate(model, model_format, device, eval_loader, class_index, batch_size):
    correct = 0.0
    class_scores = []
    class_labels = []

    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    tbar = tqdm(eval_loader)
    for i, (data, target) in enumerate(tbar):
        # support of PyTorch pth model
        if model_format == 'PTH':
            tmp_correct, class_score = predict_torch(model, device, data, target, class_index)
            correct += tmp_correct
        # support of ONNX model
        elif model_format == 'ONNX':
            tmp_correct, class_score =  predict_onnx(model, data, target, class_index)
            correct += tmp_correct
        # support of MNN model
        elif model_format == 'MNN':
            tmp_correct, class_score = predict_mnn(model, session, data, target, class_index)
            correct += tmp_correct
        else:
            raise ValueError('invalid model format')

        # record score & labels for specified class
        class_scores.append(float(class_score))
        class_labels.append(int(target.detach().cpu().numpy()))

        tbar.set_description('Evaluate acc: %06.4f' % (correct/((i + 1)*batch_size)))

    val_acc = correct / len(eval_loader.dataset)
    print('Test set accuracy: {}/{} ({:.2f})'.format(
        correct, len(eval_loader.dataset), val_acc))

    # search for a best score threshold on one class
    accuracy, threshold = threshold_search(class_scores, class_labels, class_index)
    print('Best accuracy for class[{}]({}): {:.4f}, with score threshold {:.4f}'.format(class_index, eval_loader.dataset.classes[class_index], accuracy, threshold))

    return val_acc



def load_eval_model(model_path, device):
    # support of PyTorch pth model
    if model_path.endswith('.pth'):
        model = torch.load(model_path, map_location=device)
        model_format = 'PTH'

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
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate CNN classifer model (pth/onnx/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to evaluation image dataset')

    #parser.add_argument(
        #'--classes_path', type=str, required=False,
        #help='path to class definitions, default=%(default)s', default=os.path.join('configs' , 'voc_classes.txt'))

    parser.add_argument(
        '--model_input_shape', type=str, required=False,
        help='model image input size as <height>x<width>, default=%(default)s', default='224x224')

    parser.add_argument(
        '--class_index', type=int, required=False,
        help='class index to check the best threshold, default=%(default)s', default=0)

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # prepare eval dataset loader
    eval_loader = get_dataloader(args.dataset_path, args.model_input_shape, batch_size=1, use_cuda=use_cuda, mode='eval')

    num_classes = len(eval_loader.dataset.classes)
    print('Classes:', eval_loader.dataset.classes)

    # get eval model
    model, model_format = load_eval_model(args.model_path, device)

    acc = evaluate(model, model_format, device, eval_loader, args.class_index, batch_size=1)
    #print("%.5f" % acc)


if __name__ == '__main__':
    main()
