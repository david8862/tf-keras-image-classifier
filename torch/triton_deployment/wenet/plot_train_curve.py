#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot accuracy/loss curve of WeNet training from checkpoint yamls
"""
import os, sys, argparse
import glob
import subprocess
import matplotlib.pyplot as plt


def draw_curve(checkpoint_path, output_file):
    # glob to get epoch yaml file list
    epoch_yaml_files = glob.glob(os.path.join(checkpoint_path, 'epoch_*.yaml'))
    epoch_yaml_dict = {}

    # parse epoch number from yaml file name
    for epoch_yaml_file in epoch_yaml_files:
        epoch_num = int(os.path.splitext(os.path.basename(epoch_yaml_file))[0].split('_')[-1])
        epoch_yaml_dict[epoch_num] = epoch_yaml_file

    # sort yaml files with epoch number
    sorted_epoch_yaml_list = sorted(epoch_yaml_dict.items())
    epoch_num_list = list(range(len(sorted_epoch_yaml_list)))

    epoch_acc_list = []
    epoch_loss_list = []
    for sorted_epoch_yaml_tuple in sorted_epoch_yaml_list:
        sorted_epoch_yaml_file = sorted_epoch_yaml_tuple[1] # (28, 'epoch_28.yaml')

        # grep to get epoch accuracy
        epoc_acc_proc = subprocess.Popen("grep -r 'acc:' " + sorted_epoch_yaml_file, stdout=subprocess.PIPE, shell=True)
        sorted_epoch_acc_strings = epoc_acc_proc.stdout.readlines() # [[b'  acc: 0.9372962652339009\n']]
        assert (len(sorted_epoch_acc_strings) == 1), 'invalid epoch acc'
        epoch_acc = float(sorted_epoch_acc_strings[0].strip().decode('utf-8').split(':')[-1])

        # grep to get epoch loss
        epoc_loss_proc = subprocess.Popen("grep -r ' loss:' " + sorted_epoch_yaml_file, stdout=subprocess.PIPE, shell=True)
        sorted_epoch_loss_strings = epoc_loss_proc.stdout.readlines()
        assert (len(sorted_epoch_loss_strings) == 1), 'invalid epoch loss'
        epoch_loss = float(sorted_epoch_loss_strings[0].strip().decode('utf-8').split(':')[-1])

        epoch_acc_list.append(epoch_acc)
        epoch_loss_list.append(epoch_loss)

    # prepare to plot curve
    plt.figure(figsize=(10, 10))
    plt.style.use('bmh') # bmh/ggplot/dark_background/fivethirtyeight/grayscale

    # subplot for accuracy curve
    plt.subplot(2, 1, 1)
    plt.title("Epoch accuracy")
    plt.plot(epoch_num_list, epoch_acc_list, 'o-', color='r', label="epoch accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    # subplot for loss curve
    plt.subplot(2, 1, 2)
    plt.title("Epoch loss")
    plt.plot(epoch_num_list, epoch_loss_list, 's-', color='b', label="epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # save or show result
    plt.tick_params(labelsize='medium') # set tick font size
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=75)
    else:
        plt.show()
    return


def main():
    parser = argparse.ArgumentParser(description='Plot accuracy/loss curve of WeNet training from checkpoint yamls')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='training checkpoints path which contains pt models and epoch status yamls')
    parser.add_argument('--output_file', type=str, required=False, help='output file to save curve chart image, default=%(default)s', default=None)

    args = parser.parse_args()

    draw_curve(args.checkpoint_path, args.output_file)


if __name__ == '__main__':
    main()

