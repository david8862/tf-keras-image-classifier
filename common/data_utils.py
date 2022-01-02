#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
import numpy as np
import cv2
from PIL import Image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def random_gray(x):
    prob = 0.2

    convert = rand() < prob
    if convert:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    return x


def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to -1.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, -1.0 ~ 1.0
    """
    image = image.astype(np.float32) / 127.5 - 1

    return image


def denormalize_image(image):
    """
    Denormalize image array from -1.0 ~ 1.0
    to 0 ~ 255

    # Arguments
        image: normalized image array with dtype=float, -1.0 ~ 1.0

    # Returns
        image: numpy image array with dtype=uint8, 0 ~ 255
    """
    image = (image * 127.5 + 127.5).astype(np.uint8)

    return image


def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(model_image_size, Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)
    return image_data

