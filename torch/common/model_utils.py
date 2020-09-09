#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_optimizer(optim_type, model, learning_rate):
    optim_type = optim_type.lower()

    if optim_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0, amsgrad=False)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0, momentum=0, centered=False)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0, momentum=0.9, nesterov=False)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer

