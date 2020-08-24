#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate heatmap for input images to verify
the trained CNN model
'''
import os, sys, argparse
import glob
import numpy as np
import cv2
from tensorflow.keras.models import load_model
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_custom_objects, optimize_tf_gpu
from common.preprocess_crop import load_and_crop_img

optimize_tf_gpu(tf, K)


def layer_type(layer):
    # TODO: use isinstance() instead.
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_last_conv(model):
    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]
    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        if layer == "Conv2D":
            return i

def get_target_size(model):
    if K.image_data_format() == 'channels_first':
        return model.input_shape[2:4]
    else:
        return model.input_shape[1:3]


def generate_heatmap(image_path, model_path, heatmap_path, class_names=None):
    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, custom_objects=custom_object_dict)
    K.set_learning_phase(0)
    model.summary()

    # get image file list or single image
    if os.path.isdir(image_path):
        jpeg_files = glob.glob(os.path.join(image_path, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(image_path, '*.jpg'))
        image_list = jpeg_files + jpg_files

        #assert os.path.isdir(heatmap_path), 'need to provide a path for output heatmap'
        os.makedirs(heatmap_path, exist_ok=True)
        heatmap_list = [os.path.join(heatmap_path, os.path.splitext(os.path.basename(image_name))[0]+'.jpg') for image_name in image_list]
    else:
        image_list = [image_path]
        heatmap_list = [heatmap_path]

    for i, (image_file, heatmap_file) in enumerate(zip(image_list, heatmap_list)):
        # process input
        target_size=get_target_size(model)
        x = load_and_crop_img(image_file, target_size=target_size, interpolation='nearest:center')
        x = np.array(x) / 255.
        x = np.expand_dims(x, axis=0)

        # predict and get output
        preds = model.predict(x)
        index = np.argmax(preds[0])
        print(preds[0])
        print('predict index: {}'.format(index))
        max_output = model.output[:, index]

        # detect last conv layer
        last_conv_index = detect_last_conv(model)
        last_conv_layer = model.layers[last_conv_index]
        # get gradient of the last conv layer to the predicted class
        grads = K.gradients(max_output, last_conv_layer.output)[0]
        # pooling to get the feature gradient
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        # run the predict to get value
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # apply the activation to each channel of the conv'ed feature map
        for j in range(pooled_grads_value.shape[0]):
            conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

        # get mean of each channel, which is the heatmap
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        # normalize heatmap to 0~1
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #plt.matshow(heatmap)
        #plt.show()

        # overlap heatmap to frame image
        img = cv2.imread(image_file)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        # show predict class index or name on image
        cv2.putText(superimposed_img, '{}'.format(class_names[index] if class_names else index),
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        # save overlaped image
        cv2.imwrite(heatmap_file, superimposed_img)
        print("generate heatmap file {} ({}/{})".format(heatmap_file, i+1, len(image_list)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Image file or directory to predict')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--heatmap_path', type=str, required=True, help='output heatmap file or directory')
    parser.add_argument('--classes_path', type=str, required=False, default=None, help='path to class definition, optional')

    args = parser.parse_args()

    if args.classes_path:
        class_names = get_classes(args.classes_path)
    else:
        class_names = None

    generate_heatmap(args.image_path, args.model_path, args.heatmap_path, class_names)



if __name__ == "__main__":
    main()

