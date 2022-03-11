#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50

from common.backbones.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from common.backbones.peleenet import PeleeNet
from common.backbones.ghostnet import GhostNet
from common.backbones.squeezenet import SqueezeNet
from common.backbones.mobilevit import MobileViT_S, MobileViT_XS, MobileViT_XXS
from common.backbones.simple_cnn import SimpleCNN, SimpleCNNLite


def get_base_model(model_type, model_input_shape, weights='imagenet'):

    input_tensor = Input(shape=model_input_shape+(3,), name='image_input')

    if model_type == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.5)
    elif model_type == 'mobilenetv2':
        model = MobileNetV2(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.5)
    elif model_type == 'mobilenetv3large':
        model = MobileNetV3Large(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.75)
    elif model_type == 'mobilenetv3small':
        model = MobileNetV3Small(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.75)
    elif model_type == 'peleenet':
        model = PeleeNet(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'ghostnet':
        model = GhostNet(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'squeezenet':
        model = SqueezeNet(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'mobilevit_s':
        model = MobileViT_S(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'mobilevit_xs':
        model = MobileViT_XS(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'mobilevit_xxs':
        model = MobileViT_XXS(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'resnet50':
        model = ResNet50(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False)
    elif model_type == 'simple_cnn':
        model = SimpleCNN(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=None, pooling=None, include_top=False)
    elif model_type == 'simple_cnn_lite':
        model = SimpleCNNLite(input_tensor=input_tensor, input_shape=model_input_shape+(3,), weights=None, pooling=None, include_top=False)
    else:
        raise ValueError('Unsupported model type')
    return model


def get_model(model_type, num_classes, model_input_shape, head_conv_channel, weights_path=None):
    # create the base pre-trained model
    base_model = get_base_model(model_type, model_input_shape)
    backbone_len = len(base_model.layers)

    x = base_model.output
    if head_conv_channel:
        x = Conv2D(head_conv_channel, kernel_size=1, padding='same', activation='relu', name='head_conv')(x)
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    #x = Dense(128, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(num_classes, activation='softmax', name='score_predict')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model, backbone_len

