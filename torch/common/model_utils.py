#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_lr_scheduler(decay_type, optimizer, decay_steps):
    """
    Return a learning rate scheduler
    """
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = None
    elif decay_type == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0)
    elif decay_type == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    elif decay_type == 'exponential':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif decay_type == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #decay to gamma*lr every 10 epochs
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, model, learning_rate, weight_decay):
    optim_type = optim_type.lower()

    if optim_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, momentum=0, centered=False)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=False)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer

