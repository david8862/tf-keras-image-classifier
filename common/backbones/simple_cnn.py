#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Activation, Dense, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Softmax, Reshape, Dropout, ZeroPadding2D, MaxPooling2D, SeparableConv2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def SimpleCNN(input_shape,
              weights=None,
              pooling=None,
              include_top=False,
              classes=1000,
              dropout_rate=0.2,
              l2_regularization=5e-4,
              **kwargs):

    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)

    x = ZeroPadding2D(padding=correct_pad(K, img_input, 3),
                             name='Conv_pad')(img_input)
    x = Conv2D(filters=16,
               kernel_size=3,
               strides=(2, 2),
               padding='valid',
               use_bias=False,
               kernel_regularizer=regularization,
               name='image_array', input_shape=input_shape)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2D(filters=32, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=64, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=128, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = Conv2D(filters=256, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=512, kernel_size=1, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 512))(x)
        x = Conv2D(1024,
                    kernel_size=1,
                    padding='same',
                    name='Conv_2')(x)
        x = ReLU()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Create model.
    model = Model(img_input, x, name='SimpleCNN ')
    return model


def SimpleCNNLite(input_shape,
                  weights=None,
                  pooling=None,
                  include_top=False,
                  classes=1000,
                  dropout_rate=0.2,
                  l2_regularization=5e-4,
                  **kwargs):

    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)

    x = ZeroPadding2D(padding=correct_pad(K, img_input, 3),
                             name='Conv_pad')(img_input)
    x = Conv2D(filters=16,
               kernel_size=3,
               strides=(2, 2),
               padding='valid',
               use_bias=False,
               kernel_regularizer=regularization,
               name='image_array', input_shape=input_shape)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = SeparableConv2D(filters=32, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = SeparableConv2D(filters=32, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = SeparableConv2D(filters=64, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = SeparableConv2D(filters=64, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = SeparableConv2D(filters=128, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = SeparableConv2D(filters=128, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = SeparableConv2D(filters=256, kernel_size=3, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=correct_pad(K, x, 3))(x)
    x = SeparableConv2D(filters=256, kernel_size=3, strides=(2, 2), kernel_regularizer=regularization, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=512, kernel_size=1, kernel_regularizer=regularization, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 512))(x)
        x = Conv2D(1024,
                    kernel_size=1,
                    padding='same',
                    name='Conv_2')(x)
        x = ReLU()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Create model.
    model = Model(img_input, x, name='SimpleCNN ')
    return model



#if __name__ == "__main__":
    #input_shape = (64, 64, 1)
    #num_classes = 7
    #model = tiny_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = mini_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = big_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = simple_CNN((48, 48, 1), num_classes)
    #model.summary()
