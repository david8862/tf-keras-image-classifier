#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import preprocess_crop


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def random_gray(x):
    prob = 2.0

    convert = rand() < prob
    if convert:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    return x


def get_data_generator(data_path, model_input_shape, batch_size, class_names, mode='train'):
    if mode == 'train':
        datagen = ImageDataGenerator(
            preprocessing_function=random_gray,
            rescale=1./255,
            #samplewise_std_normalization=True,
            #validation_split=0.1,
            zoom_range=0.25,
            brightness_range=[0.5,1.5],
            channel_shift_range=0.1,
            shear_range=0.2,
            vertical_flip=True,
            horizontal_flip=True,
            #rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='constant',
            cval=0.)

        data_generator = datagen.flow_from_directory(
            data_path,
            target_size=model_input_shape,
            batch_size=batch_size,
            classes=class_names,
            class_mode='categorical',
            #save_to_dir='check',
            #save_prefix='augmented_',
            #save_format='jpg',
            interpolation = 'nearest:center')

    elif mode == 'val' or mode == 'eval':
        datagen = ImageDataGenerator(
            preprocessing_function=random_gray,
            rescale=1./255,
            #samplewise_std_normalization=True)
            )

        data_generator = datagen.flow_from_directory(
            data_path,
            target_size=model_input_shape,
            batch_size=batch_size,
            classes=class_names,
            class_mode='categorical',
            interpolation = 'nearest')
    else:
        raise ValueError('output model file is not specified')

    return data_generator

