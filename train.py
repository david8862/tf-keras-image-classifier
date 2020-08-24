#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
import os, sys, argparse, time
import numpy as np
import cv2

from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow.keras.backend as K
import tensorflow as tf

from common import preprocess_crop
from common.utils import get_classes, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.backbones.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from common.backbones.simple_cnn import SimpleCNN, SimpleCNNLite

optimize_tf_gpu(tf, K)



def get_generators(train_data_path, val_data_path, model_input_shape, batch_size, class_names):
    train_datagen = ImageDataGenerator(
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

    val_datagen = ImageDataGenerator(rescale=1./255)
        #samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=model_input_shape,
        batch_size=batch_size,
        classes=class_names,
        class_mode='categorical',
        #save_to_dir='check',
        #save_prefix='augmented_',
        #save_format='jpg',
        interpolation = 'nearest:center')

    val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=model_input_shape,
        batch_size=batch_size,
        classes=class_names,
        class_mode='categorical',
        interpolation = 'nearest:center')

    return train_generator, val_generator


def get_base_model(model_type, model_input_shape, weights='imagenet'):
    if model_type == 'mobilenet':
        model = MobileNet(input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.5)
    elif model_type == 'mobilenetv2':
        model = MobileNetV2(input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.5)
    elif model_type == 'mobilenetv3large':
        model = MobileNetV3Large(input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.75)
    elif model_type == 'mobilenetv3small':
        model = MobileNetV3Small(input_shape=model_input_shape+(3,), weights=weights, pooling=None, include_top=False, alpha=0.75)
    elif model_type == 'simple_cnn':
        model = SimpleCNN(input_shape=model_input_shape+(3,), weights=None, pooling=None, include_top=False)
    elif model_type == 'simple_cnn_lite':
        model = SimpleCNNLite(input_shape=model_input_shape+(3,), weights=None, pooling=None, include_top=False)
    else:
        raise ValueError('Unsupported model type')
    return model


def get_model(model_type, class_names, model_input_shape, head_conv_channel, weights_path=None):
    # create the base pre-trained model
    base_model = get_base_model(model_type, model_input_shape)
    backbone_len = len(base_model.layers)

    # add a global spatial average pooling layer
    #x = base_model.get_layer('conv_pw_11_relu').output
    x = base_model.output

    if head_conv_channel:
        x = Conv2D(head_conv_channel, kernel_size=1, padding='same', activation='relu', name='head_conv')(x)
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    #x = Dense(128, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(len(class_names), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model, backbone_len



def main(args):
    log_dir = 'logs'
    # get class info
    class_names = get_classes(args.classes_path)

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'),
        monitor='val_acc',
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.5, patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', min_delta=0, patience=50, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    #learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    #lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

    callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

    # prepare train&val data generator
    train_generator, val_generator = get_generators(args.train_data_path, args.val_data_path, args.model_input_shape, args.batch_size, class_names)

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=None)

    # get train model
    model, backbone_len = get_model(args.model_type, class_names, args.model_input_shape, args.head_conv_channel, args.weights_path)
    model.summary()

    # Freeze backbone part for transfer learning
    for i in range(backbone_len):
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(backbone_len, len(model.layers)))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch
    epochs = initial_epoch + args.transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(train_generator.samples, val_generator.samples, args.batch_size, args.model_input_shape))
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // args.batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // args.batch_size,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        #verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks=callbacks)

    # Wait 2 seconds for next stage
    time.sleep(2)

    if args.decay_type:
        # rebuild optimizer to apply learning rate decay, only after
        # unfreeze all layers
        callbacks.remove(reduce_lr)
        steps_per_epoch = max(1, train_generator.samples//args.batch_size)
        decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
        optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=args.decay_type, decay_steps=decay_steps)


    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    print("Unfreeze and continue training, to fine-tune.")
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // args.batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // args.batch_size,
                        epochs=args.total_epoch,
                        initial_epoch=epochs,
                        #verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks=callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='train a simple CNN classifier')
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='mobilenetv2',
        help='backbone model type: mobilenetv2/mobilenet(v1), default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='224x224',
        help = "model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--head_conv_channel', type=int, required=False, default=None,
        help = "channel number for head part convolution, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_path', type=str, required=True,
        help='path to train data')
    parser.add_argument('--val_data_path', type=str, required=True,
        help='path to validation dataset')
    parser.add_argument('--classes_path', type=str, required=True,
        help='path to classes definition', default=None)

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=32,
        help = "batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")

    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--transfer_epoch', type=int, required=False, default=20,
        help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=100,
        help = "Total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')


    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    main(args)
