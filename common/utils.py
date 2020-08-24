#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Miscellaneous utility functions."""

import os
import numpy as np
import time
import cv2, colorsys
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from common.backbones.mobilenet_v3 import hard_sigmoid, hard_swish
import tensorflow as tf


def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
                    #tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
        session = tf.Session(config=config)

        # set session
        K.set_session(session)


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
    }

    return custom_objects_dict


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image

