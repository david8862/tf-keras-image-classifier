#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from classifier.model import Classifier
from classifier.data import get_dataloader
from common.model_utils import get_optimizer, get_lr_scheduler


# global value to record the best accuracy
best_acc = 0.0


def checkpoint_clean(checkpoint_dir, max_keep=5):
    # filter out checkpoints and sort
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'ep*.pth')), reverse=False)

    # keep latest checkpoints
    for checkpoint in checkpoints[:-(max_keep)]:
        os.remove(checkpoint)


def train(args, epoch, model, device, num_classes, train_loader, optimizer, lr_scheduler, summary_writer):
    epoch_loss = 0.0
    epoch_correct = 0.0
    epoch_topk_correct = 0.0

    model.train()
    tbar = tqdm(train_loader)
    for i, (data, target) in enumerate(tbar):
        # forward propagation
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # calculate loss
        #loss = F.cross_entropy(output, target)
        #loss = F.nll_loss(torch.log(output), target)
        #loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.NLLLoss()(torch.log(output), target)

        # backward propagation
        loss.backward()
        optimizer.step()

        # collect loss and accuracy
        batch_loss = loss.item()
        epoch_loss += batch_loss

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        epoch_correct += batch_correct

        if num_classes > 10:
            # only collect top 5 accuracy when more than 10 class
            _, pred = output.topk(5, dim=1, largest=True, sorted=True)
            target_resize = target.view(-1, 1)
            batch_topk_correct = pred.eq(target_resize).sum().item()
            epoch_topk_correct += batch_topk_correct
            tbar.set_description('Train loss: %06.4f - acc: %06.4f - topk acc: %06.4f' % (epoch_loss/(i + 1), epoch_correct/((i + 1)*args.batch_size), epoch_topk_correct/((i + 1)*args.batch_size)))

        else:
            tbar.set_description('Train loss: %06.4f - acc: %06.4f' % (epoch_loss/(i + 1), epoch_correct/((i + 1)*args.batch_size)))

        # log train loss and accuracy
        summary_writer.add_scalar('train loss', batch_loss, epoch*len(train_loader)+i)
        summary_writer.add_scalar('train accuracy', batch_correct/args.batch_size, epoch*len(train_loader)+i)
        if num_classes > 10:
            summary_writer.add_scalar('train topk accuracy', batch_topk_correct/args.batch_size, epoch*len(train_loader)+i)


    epoch_loss /= len(train_loader)
    # decay learning rate every epoch
    if lr_scheduler:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(epoch_loss)
        else:
            lr_scheduler.step()


def validate(args, epoch, step, model, device, num_classes, val_loader, log_dir, summary_writer):
    global best_acc
    val_loss = 0.0
    correct = 0.0
    topk_correct = 0.0

    model.eval()
    with torch.no_grad():
        tbar = tqdm(val_loader)
        #for data, target in val_loader:
        for i, (data, target) in enumerate(tbar):
            # inference on validate data
            data, target = data.to(device), target.to(device)
            output = model(data)

            # collect loss and accuracy
            #val_loss += F.cross_entropy(output, target, reduction='sum').item() / args.batch_size # sum up batch loss
            val_loss += F.nll_loss(torch.log(output), target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if num_classes > 10:
                # only collect top 5 accuracy when more than 10 class
                _, pred = output.topk(5, dim=1, largest=True, sorted=True)
                target_resize = target.view(-1, 1)
                topk_correct += pred.eq(target_resize).sum().item()
                tbar.set_description('Validate loss: %06.4f - acc: %06.4f - topk acc: %06.4f' % (val_loss/((i + 1)*args.batch_size), correct/((i + 1)*args.batch_size), topk_correct/((i + 1)*args.batch_size)))
            else:
                tbar.set_description('Validate loss: %06.4f - acc: %06.4f' % (val_loss/((i + 1)*args.batch_size), correct/((i + 1)*args.batch_size)))

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    # log validation loss and accuracy
    summary_writer.add_scalar('val loss', val_loss, step)
    summary_writer.add_scalar('val accuracy', val_acc, step)

    if num_classes > 10:
        val_topk_acc = topk_correct / len(val_loader.dataset)
        summary_writer.add_scalar('val topk accuracy', val_topk_acc, step)
        print('Validate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Topk Accuracy: {}/{} ({:.2f}%)'.format(
            val_loss, correct, len(val_loader.dataset), val_acc, topk_correct, len(val_loader.dataset), val_topk_acc))
    else:
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
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'plateau', 'exponential', 'step'],
        help = "Learning rate decay type, default=%(default)s")
    parser.add_argument('--weight_decay', type=float, required=False, default=5e-4,
        help = "Weight decay for optimizer, default=%(default)s")

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

    # HBONet has some input shape limitation
    if args.model_type == 'hbonet':
        assert (args.model_input_shape[0]%32 == 0 and args.model_input_shape[1]%32 == 0), 'for hbonet, model_input_shape should be multiples of 32'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # prepare train&val dataset loader
    train_loader = get_dataloader(args.train_data_path, args.model_input_shape, args.batch_size, use_cuda=use_cuda, mode='train')
    val_loader = get_dataloader(args.val_data_path, args.model_input_shape, args.batch_size, use_cuda=use_cuda, mode='val')

    # get tensorboard summary writer
    summary_writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    # check if classes match on train & val dataset
    assert train_loader.dataset.classes == val_loader.dataset.classes, 'class mismatch between train & val dataset'
    num_classes = len(train_loader.dataset.classes)
    print('Classes:', train_loader.dataset.classes)

    # get train model
    model = Classifier(args.model_type, num_classes, args.head_conv_channel).to(device)
    summary(model, input_size=(3,)+args.model_input_shape)

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
        print('Load weights {}.'.format(args.weights_path))

    optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay)

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
        train(args, epoch, model, device, num_classes, train_loader, optimizer, None, summary_writer)
        validate(args, epoch, epoch*len(train_loader), model, device, num_classes, val_loader, log_dir, summary_writer)
        checkpoint_clean(log_dir, max_keep=5)


    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

    # apply learning rate decay only after unfreeze all layers
    # NOTE: PyTorch apply learning rate scheduler for every epoch, not batch
    #steps_per_epoch = max(1, len(train_loader.dataset)//args.batch_size)
    #decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
    decay_steps = args.total_epoch - args.init_epoch - args.transfer_epoch
    lr_scheduler = get_lr_scheduler(args.decay_type, optimizer, decay_steps)

    # Fine tune train loop
    for epoch in range(epochs, args.total_epoch):
        print('Epoch %d/%d'%(epoch, args.total_epoch))
        train(args, epoch, model, device, num_classes, train_loader, optimizer, lr_scheduler, summary_writer)
        validate(args, epoch, epoch*len(train_loader), model, device, num_classes, val_loader, log_dir, summary_writer)
        checkpoint_clean(log_dir, max_keep=5)

    # Finally store model
    torch.save(model, os.path.join(log_dir, 'trained_final.pth'))
    #torch.save(model.state_dict(), os.path.join(log_dir, 'trained_final.pt'))

if __name__ == '__main__':
    main()
