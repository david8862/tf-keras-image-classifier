#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from classifier.model import Classifier
from classifier.data import get_dataloader
from common.model_utils import get_optimizer


# global value to record the best accuracy
best_acc = 0.0


def train(args, model, device, train_loader, optimizer):
    train_loss = 0.0
    correct = 0.0
    model.train()
    tbar = tqdm(train_loader)
    for i, (data, target) in enumerate(tbar):
        # forward propagation
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # calculate loss
        loss = F.cross_entropy(output, target)
        #loss = F.nll_loss(output, target)

        # backward propagation
        loss.backward()
        optimizer.step()

        # collect loss and accuracy
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        tbar.set_description('Train loss: %06.4f - acc: %06.4f' % (train_loss/(i + 1), correct/((i + 1)*args.batch_size)))



def validate(args, model, device, val_loader, epoch, log_dir):
    global best_acc
    val_loss = 0.0
    correct = 0.0
    model.eval()
    with torch.no_grad():
        tbar = tqdm(val_loader)
        #for data, target in val_loader:
        for i, (data, target) in enumerate(tbar):
            # inference on validate data
            data, target = data.to(device), target.to(device)
            output = model(data)

            # collect loss and accuracy
            val_loss += F.cross_entropy(output, target, reduction='sum').item() / args.batch_size # sum up batch loss
            #val_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            tbar.set_description('Validate loss: %06.4f - acc: %06.4f' % (val_loss/(i + 1), correct/((i + 1)*args.batch_size)))

    val_loss /= (len(val_loader.dataset) / args.batch_size)
    val_acc = correct / len(val_loader.dataset)
    print('Validate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset), val_acc))

    # save checkpoint with best accuracy
    if val_acc > best_acc:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.pth'.format(epoch=epoch+1, val_loss=val_loss, val_acc=val_acc))
        torch.save(model, checkpoint_dir)
        print('Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch+1, best_acc=best_acc, val_acc=val_acc, checkpoint_dir=checkpoint_dir))
        best_acc = val_acc
    else:
        print('Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}'.format(epoch=epoch+1, best_acc=best_acc))



def main():
    parser = argparse.ArgumentParser(description='train a simple CNN classifier with PyTorch')
    log_dir = os.path.join('logs', '000')

    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='mobilenetv2',
        help='backbone model type: mobilenetv3/v2/simple_cnn, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='224x224',
        help = "model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--head_conv_channel', type=int, required=False, default=128,
        help = "channel number for head part convolution, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='path to train image dataset')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='path to validation image dataset')

    # Training settings
    parser.add_argument('--batch_size', type=int, required=False, default=64,
        help = "batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")

    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--transfer_epoch', type=int, required=False, default=5,
        help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=100,
        help = "Total training epochs, default=%(default)s")
    #parser.add_argument('--gpu_num', type=int, required=False, default=1,
        #help='Number of GPU to use, default=%(default)s')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # prepare train&val dataset loader
    train_loader = get_dataloader(args.train_data_path, args.model_input_shape, args.batch_size, use_cuda=use_cuda, mode='train')
    val_loader = get_dataloader(args.val_data_path, args.model_input_shape, args.batch_size, use_cuda=use_cuda, mode='val')

    # check if classes match on train & val dataset
    assert train_loader.dataset.classes == val_loader.dataset.classes, 'class mismatch between train & val dataset'
    num_classes = len(train_loader.dataset.classes)
    print('Classes:', train_loader.dataset.classes)

    # get train model
    model = Classifier(args.model_type, num_classes, args.head_conv_channel).to(device)
    summary(model, input_size=(3,)+args.model_input_shape)

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))
        print('Load weights {}.'.format(args.weights_path))


    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = get_optimizer('sgd', model, args.learning_rate)


    # Freeze feature extractor part for transfer learning
    print('Freeze feature extractor part.')
    for child in model.features.children():
        for param in child.parameters():
            param.requires_grad = False


    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch
    epochs = args.init_epoch + args.transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(len(train_loader.dataset), len(val_loader.dataset), args.batch_size, args.model_input_shape))


    # Transfer train loop
    for epoch in range(initial_epoch, epochs):
        print('Epoch %d/%d'%(epoch, epochs))
        train(args, model, device, train_loader, optimizer)
        validate(args, model, device, val_loader, epoch, log_dir)


    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


    # Fine tune train loop
    for epoch in range(epochs, args.total_epoch):
        print('Epoch %d/%d'%(epoch, args.total_epoch))
        train(args, model, device, train_loader, optimizer)
        validate(args, model, device, val_loader, epoch, log_dir)

    # Finally store model
    torch.save(model, os.path.join(log_dir, 'trained_final.pt'))
    #torch.save(model.state_dict(), os.path.join(log_dir, 'trained_final.pt'))

if __name__ == '__main__':
    main()