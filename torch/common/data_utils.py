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

