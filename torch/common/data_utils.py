#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
from PIL import Image
from classifier.data import get_transform


def preprocess_image(image, target_size, return_tensor=False):
    """
    Prepare model input image data with
    preprocess transform

    # Arguments
        image: origin input image
            PIL Image object containing image data
        target_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array or PyTorch tensor
                    of image data for model input.
    """
    # load & preprocess image
    transform = get_transform(target_size, mode='eval')

    image_tensor = transform(image)

    if return_tensor:
        image_data = image_tensor
    else:
        image_data = image_tensor.detach().cpu().numpy()

    return image_data


def denormalize_image(image, return_tensor=False):
    """
    Denormalize image tensor from (-1, 1) to (0, 255)

    # Arguments
        image: normalized image array or tensor
            distribution (-1, 1)
        return_tensor: whether return a PyTorch tensor
            or numpy array.

    # Returns
        image_data: numpy array or PyTorch tensor
                    of image data with distribution (0, 255).
    """
    if return_tensor:
        image_data = (image * 127.5 + 127.5).byte()
    else:
        image_data = (image * 127.5 + 127.5).astype(np.uint8)

    return image_data
