#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate mAP for YOLO model on some annotation dataset
"""
import os, argparse, time
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf

from classifier.data import get_data_generator
from common.utils import get_classes, optimize_tf_gpu, get_custom_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)



def evaluate_accuracy(args, model, class_names):
    batch_size = 1
    # eval data generator
    eval_generator = get_data_generator(args.dataset_path, args.model_input_shape, batch_size, class_names, mode='eval')

    # start evaluation
    model.compile(
              optimizer='adam',
              metrics=['accuracy', 'top_k_categorical_accuracy'],
              loss='categorical_crossentropy')

    print('Evaluate on {} samples, with batch size {}.'.format(eval_generator.samples, batch_size))
    scores = model.evaluate_generator(
            eval_generator,
            steps=eval_generator.samples // batch_size,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            verbose=1)

    print('Evaluate loss:', scores[0])
    print('Top-1 accuracy:', scores[1])
    print('Top-k accuracy:', scores[2])



def load_eval_model(model_path):
    # normal keras h5 model
    if model_path.endswith('.h5'):
        custom_object_dict = get_custom_objects()

        model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
        model_format = 'H5'
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model, model_format


def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate CNN classifer model (h5) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to evaluation image dataset')

    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to class definitions, default=%(default)s', default=os.path.join('configs' , 'voc_classes.txt'))

    parser.add_argument(
        '--model_input_shape', type=str,
        help='model image input size as <height>x<width>, default=%(default)s', default='224x224')

    args = parser.parse_args()

    # param parse
    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    model, model_format = load_eval_model(args.model_path)

    start = time.time()
    evaluate_accuracy(args, model, class_names)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
