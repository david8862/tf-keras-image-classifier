#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
from torchsummary import summary

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image, denormalize_image
from common.utils import get_classes


def generate_heatmap(image_path, model_path, model_input_shape, heatmap_path, class_names=None):
    # hook function for getting gradient
    def extract(g):
        global features_grad
        features_grad = g

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    summary(model, input_size=(3,)+model_input_shape)

    # get image file list or single image
    if os.path.isdir(image_path):
        jpeg_files = glob.glob(os.path.join(image_path, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(image_path, '*.jpg'))
        image_list = jpeg_files + jpg_files

        os.makedirs(heatmap_path, exist_ok=True)
        heatmap_list = [os.path.join(heatmap_path, os.path.splitext(os.path.basename(image_name))[0]+'.jpg') for image_name in image_list]
    else:
        image_list = [image_path]
        heatmap_list = [heatmap_path]


    # loop the sample list to generate all heatmaps
    for i, (image_file, heatmap_file) in enumerate(zip(image_list, heatmap_list)):
        # process input
        image = Image.open(image_file).convert('RGB')
        img = preprocess_image(image, target_size=model_input_shape, return_tensor=True)
        x = img.unsqueeze(0).to(device)

        # got feature map & predict score of the model
        model.eval()
        features = model.features(x)
        output = model.classifier(features)
        feature_channel = list(features.shape)[1] # feature map channel number

        # got class index with highest predict score
        index = torch.argmax(output).item()
        pred_class = output[:, index]
        score = pred_class.detach().item()

        features.register_hook(extract)
        # get gradient of the feature map to the predicted class
        pred_class.backward()
        grads = features_grad # get gradient from global value in hook function
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)) # pooling to get the feature gradient value

        # here batch_size=1, so we get ride of that
        pooled_grads = pooled_grads[0]
        features = features[0]

        # apply the activation to each channel of the conv'ed feature map
        for j in range(feature_channel):
            features[j, ...] *= pooled_grads[j, ...]

        # get mean of each channel, which is the heatmap
        heatmap = features.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        # normalize heatmap to 0~1
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #plt.matshow(heatmap)
        #plt.show()

        # overlap heatmap to frame image
        #img = cv2.imread(image_file)
        img = img.detach().cpu().numpy()
        # convert pytorch channel first tensor back
        # to channel last
        img = np.moveaxis(img, 0, -1)

        # De-normalize tensor to image, and resize
        # for result display
        img = denormalize_image(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        # show predict class index or name on image
        cv2.putText(superimposed_img, '{name}:{conf:.3f}'.format(name=class_names[index] if class_names else index, conf=float(score)),
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        # save overlaped image
        cv2.imwrite(heatmap_file, superimposed_img)
        print("generate heatmap file {} ({}/{})".format(heatmap_file, i+1, len(image_list)))



def main():
    parser = argparse.ArgumentParser(description='check heatmap activation for CNN classifer model (pth) with test images')
    parser.add_argument('--image_path', type=str, required=True, help='Image file or directory to predict')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--heatmap_path', type=str, required=True, help='output heatmap file or directory')
    parser.add_argument('--classes_path', type=str, required=False, default=None, help='path to class definition, optional')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    if args.classes_path:
        class_names = get_classes(args.classes_path)
    else:
        class_names = None

    generate_heatmap(args.image_path, args.model_path, args.model_input_shape, args.heatmap_path, class_names)


if __name__ == "__main__":
    main()

