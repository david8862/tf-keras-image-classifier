#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
import os, sys, argparse, time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
import tensorflow.keras.backend as K
import tensorflow as tf

from classifier.model import get_model
from classifier.data import get_data_generator
from common.utils import get_classes, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import CheckpointCleanCallBack

optimize_tf_gpu(tf, K)


def main(args):
    log_dir = 'logs/000'

    # get class info
    if args.classes_path:
        class_names = get_classes(args.classes_path)
    else:
        class_names = None

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
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_keep=5)
    #learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    #lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan, checkpoint_clean]

    # prepare train&val data generator
    train_generator = get_data_generator(args.train_data_path, args.model_input_shape, args.batch_size, class_names, mode='train')
    val_generator = get_data_generator(args.val_data_path, args.model_input_shape, args.batch_size, class_names, mode='val')

    # check if classes match on train & val dataset
    assert train_generator.class_indices == val_generator.class_indices, 'class mismatch between train & val dataset'
    if not class_names:
        class_names = list(train_generator.class_indices.keys())
    print('Classes:', class_names)

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=None)

    # get train model
    model, backbone_len = get_model(args.model_type, len(class_names), args.model_input_shape, args.head_conv_channel, args.weights_path)
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
        optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=args.decay_type, decay_steps=decay_steps)


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
        help='backbone model type: mobilenetv3/v2/simple_cnn, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='224x224',
        help = "model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--head_conv_channel', type=int, required=False, default=128,
        help = "channel number for head part convolution, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_path', type=str, required=True,
        help='path to train image dataset')
    parser.add_argument('--val_data_path', type=str, required=True,
        help='path to validation image dataset')
    parser.add_argument('--classes_path', type=str, required=False,
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
    parser.add_argument('--transfer_epoch', type=int, required=False, default=5,
        help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=100,
        help = "Total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    main(args)
