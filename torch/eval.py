#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse, time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

import torch
import MNN
import onnxruntime

from classifier.data import get_dataloader


def topk_np(matrix, K, axis=1):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]

    return topk_data_sort, topk_index_sort


def predict_torch(model, device, num_classes, data, target, class_index):
    model.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        if num_classes > 10:
            # only collect top 5 accuracy when more than 10 class
            _, tmp_pred = output.topk(5, dim=1, largest=True, sorted=True)
            target_resize = target.view(-1, 1)
            topk_correct = tmp_pred.eq(target_resize).sum().item()
        else:
            topk_correct = 0.0

        class_score = output.detach().cpu().numpy()[:, class_index]
    return pred.detach().cpu().numpy(), topk_correct, class_score


def predict_onnx(model, num_classes, data, target, class_index):
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

    if num_classes > 10:
        # only collect top 5 accuracy when more than 10 class
        _, tmp_pred = topk_np(output[0], 5, axis=1)
        target_resize = target.reshape(-1, 1)
        topk_correct = float(np.equal(tmp_pred, target_resize).astype(np.int32).sum())
    else:
        topk_correct = 0.0

    class_score = output[0][:, class_index]
    return pred, topk_correct, class_score


def predict_mnn(interpreter, session, num_classes, data, target, class_index):
    from functools import reduce
    from operator import mul

    # convert pytorch tensor to numpy array
    data, target = data.detach().cpu().numpy(), target.detach().cpu().numpy()

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
    # use Caffe NCHW layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, channel, height, width)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Caffe)

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

    if num_classes > 10:
        # only collect top 5 accuracy when more than 10 class
        _, tmp_pred = topk_np(output[0], 5, axis=1)
        target_resize = target.reshape(-1, 1)
        topk_correct = float(np.equal(tmp_pred, target_resize).astype(np.int32).sum())
    else:
        topk_correct = 0.0

    class_score = output_data[:, class_index]
    return pred, topk_correct, class_score


def plot_confusion_matrix(cm, classes, accuracy, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
    trained_classes = classes
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.ylabel('True label', fontsize=9)
    plt.xlabel('Predicted label', fontsize=9)

    plt.title(title + '\nAccuracy: ' + str(np.round(accuracy*100, 2)))
    output_path = os.path.join('result', 'confusion_matrix.png')
    os.makedirs('result', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    #plt.show()

    # close the plot
    plt.close()
    return


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


def evaluate(model, model_format, device, class_names, eval_loader, class_index, batch_size):
    correct = 0.0
    topk_correct = 0.0
    num_classes = len(class_names)

    target_list = []
    pred_list = []
    class_scores = []
    class_labels = []

    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    tbar = tqdm(eval_loader)
    for i, (data, target) in enumerate(tbar):
        # support of PyTorch pth model
        if model_format == 'PTH':
            pred, tmp_topk_correct, class_score = predict_torch(model, device, num_classes, data, target, class_index)
            topk_correct += tmp_topk_correct
        # support of ONNX model
        elif model_format == 'ONNX':
            pred, tmp_topk_correct, class_score = predict_onnx(model, num_classes, data, target, class_index)
            topk_correct += tmp_topk_correct
        # support of MNN model
        elif model_format == 'MNN':
            pred, tmp_topk_correct, class_score = predict_mnn(model, session, num_classes, data, target, class_index)
            topk_correct += tmp_topk_correct
        else:
            raise ValueError('invalid model format')

        target = target.detach().cpu().numpy()
        correct += float(np.equal(pred, target).astype(np.int32).sum())

        # record pred & target for confusion matrix
        target_list.append(target)
        pred_list.append(pred)

        # record score & labels for specified class
        class_scores.append(float(class_score))
        class_labels.append(int(target))

        if num_classes > 10:
            tbar.set_description('Evaluate acc: %06.4f, topk acc: %06.4f' % (correct/((i + 1)*batch_size), topk_correct/((i + 1)*batch_size)))
        else:
            tbar.set_description('Evaluate acc: %06.4f' % (correct/((i + 1)*batch_size)))

    val_acc = correct / len(eval_loader.dataset)
    print('Test set accuracy: {}/{} ({:.2f})'.format(
        correct, len(eval_loader.dataset), val_acc))

    # Plot accuracy & confusion matrix
    confusion_mat = confusion_matrix(y_true=np.squeeze(target_list), y_pred=np.squeeze(pred_list), labels=list(range(len(class_names))))
    plot_confusion_matrix(confusion_mat, class_names, val_acc, normalize=True)

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
    batch_size = 1
    eval_loader = get_dataloader(args.dataset_path, args.model_input_shape, batch_size=batch_size, use_cuda=use_cuda, mode='eval')

    class_names = eval_loader.dataset.classes
    print('Classes:', class_names)

    # get eval model
    model, model_format = load_eval_model(args.model_path, device)

    start = time.time()
    acc = evaluate(model, model_format, device, class_names, eval_loader, args.class_index, batch_size=batch_size)
    #print("%.5f" % acc)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
