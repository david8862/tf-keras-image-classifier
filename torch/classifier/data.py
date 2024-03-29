#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import torch
from torchvision import datasets, transforms

def get_transform(target_size, mode='train'):

    if mode == 'train':
        transform=transforms.Compose([
                               transforms.Resize(target_size),
                               #transforms.CenterCrop(target_size),
                               #transforms.RandomCrop(target_size, padding=0, pad_if_needed=True),
                               #transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=2),

                               transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
                               #transforms.RandomApply([transforms.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.2),
                               transforms.RandomGrayscale(p=0.1),
                               #transforms.Grayscale(num_output_channels=3),

                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               #transforms.RandomRotation(30, resample=False, expand=False, center=None),

                               transforms.ToTensor(), # normalize from (0, 255) to (0, 1)
                               transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                               transforms.Normalize((0.5,), (0.5,)) # normalize from (0, 1) to (-1, 1)
                           ])
    elif mode == 'val' or mode == 'eval':
        transform=transforms.Compose([
                               transforms.Resize(target_size),
                               #transforms.CenterCrop(target_size),
                               #transforms.RandomCrop(target_size, padding=0, pad_if_needed=True),
                               #transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor(), # normalize from (0, 255) to (0, 1)
                               transforms.Normalize((0.5,), (0.5,)) # normalize from (0, 1) to (-1, 1)
                           ])
    else:
        raise ValueError('Unsupported mode ', mode)

    return transform


def get_dataloader(data_path, target_size, batch_size, use_cuda, mode='train'):

    transform = get_transform(target_size, mode=mode)

    # prepare dataset loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = datasets.ImageFolder(data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader

