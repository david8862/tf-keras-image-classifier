#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, glob
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import OrderedDict
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Input path for the converted image', type=str)
    parser.add_argument('--output_path', help='Output path for the converted image', type=str)
    args = parser.parse_args()

    jpeg_files = glob.glob(os.path.join(args.input_path, '*.jpg'))

    # create output path
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(total=len(jpeg_files), desc='Getting ROI')
    for i, jpeg_file in enumerate(jpeg_files):

        img = cv2.imread(jpeg_file, cv2.IMREAD_COLOR)
        jpeg_basename = jpeg_file.split(os.path.sep)[-1].split('.')[-2]

        ymin = int(img.shape[0]*0.33)
        ymax = int(img.shape[0]*0.66)

        xmin = 0
        xmax = int(img.shape[1]*0.25)

        area = img[ymin:ymax, xmin:xmax]
        output_img_name = os.path.join(args.output_path, '%s_0.jpg'%(jpeg_basename))
        cv2.imwrite(output_img_name, area)


        xmin = int(img.shape[1]*0.25)
        xmax = int(img.shape[1]*0.5)

        area = img[ymin:ymax, xmin:xmax]
        output_img_name = os.path.join(args.output_path, '%s_1.jpg'%(jpeg_basename))
        cv2.imwrite(output_img_name, area)

        xmin = int(img.shape[1]*0.5)
        xmax = int(img.shape[1]*0.75)

        area = img[ymin:ymax, xmin:xmax]
        output_img_name = os.path.join(args.output_path, '%s_2.jpg'%(jpeg_basename))
        cv2.imwrite(output_img_name, area)

        xmin = int(img.shape[1]*0.75)
        xmax = int(img.shape[1])

        area = img[ymin:ymax, xmin:xmax]
        output_img_name = os.path.join(args.output_path, '%s_3.jpg'%(jpeg_basename))
        cv2.imwrite(output_img_name, area)
        pbar.update(1)
    pbar.close()



if __name__ == "__main__":
    main()

