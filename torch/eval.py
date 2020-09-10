#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from classifier.data import get_dataloader


def evaluate(model, device, eval_loader, batch_size):
    val_loss = 0.0
    correct = 0.0
    model.eval()
    with torch.no_grad():
        tbar = tqdm(eval_loader)
        for i, (data, target) in enumerate(tbar):
            # inference on validate data
            data, target = data.to(device), target.to(device)
            output = model(data)

            # collect loss and accuracy
            val_loss += F.cross_entropy(output, target, reduction='sum').item() / batch_size # sum up batch loss
            #val_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            tbar.set_description('Evaluate loss: %06.4f - acc: %06.4f' % (val_loss/(i + 1), correct/((i + 1)*batch_size)))

    val_loss /= (len(eval_loader.dataset) / batch_size)
    val_acc = correct / len(eval_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(eval_loader.dataset), val_acc))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate CNN classifer model (pth) with test dataset')
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

    # get train model
    model = torch.load(args.model_path, map_location=device)
    summary(model, input_size=(3,)+args.model_input_shape)

    evaluate(model, device, eval_loader, batch_size=1)


if __name__ == '__main__':
    main()
