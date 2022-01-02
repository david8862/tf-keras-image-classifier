#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import preprocess_crop
from common.data_utils import normalize_image, random_grayscale, random_chroma, random_contrast, random_sharpness


def preprocess(image):
    # random adjust color level
    image = random_chroma(image)

    # random adjust contrast
    image = random_contrast(image)

    # random adjust sharpness
    image = random_sharpness(image)

    # random convert image to grayscale
    image = random_grayscale(image)

    # normalize image
    image = normalize_image(image)

    return image


def get_data_generator(data_path, model_input_shape, batch_size, class_names, mode='train'):
    if mode == 'train':
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            #featurewise_center=False,
            #samplewise_center=False,
            #featurewise_std_normalization=False,
            #samplewise_std_normalization=False,
            #zca_whitening=False,
            #zca_epsilon=1e-06,
            zoom_range=0.25,
            brightness_range=[0.5,1.5],
            channel_shift_range=0.1,
            shear_range=0.2,
            rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=True,
            horizontal_flip=True,
            #rescale=1./255,
            #validation_split=0.1,
            fill_mode='constant',
            cval=0.,
            data_format=None,
            dtype=None)

        data_generator = datagen.flow_from_directory(
            data_path,
            target_size=model_input_shape,
            batch_size=batch_size,
            color_mode='rgb',
            classes=class_names,
            class_mode='categorical',
            shuffle=True,
            #save_to_dir='check',
            #save_prefix='augmented_',
            #save_format='jpg',
            interpolation='nearest:center')

    elif mode == 'val' or mode == 'eval':
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            #rescale=1./255,
            )

        data_generator = datagen.flow_from_directory(
            data_path,
            target_size=model_input_shape,
            batch_size=batch_size,
            color_mode='rgb',
            classes=class_names,
            class_mode='categorical',
            shuffle=True,
            interpolation='nearest')
    else:
        raise ValueError('output model file is not specified')

    return data_generator

