#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Input path for the converted image', type=str)
    parser.add_argument('--output_path', help='Output path for the converted image', type=str)
    args = parser.parse_args()

    jpeg_files = glob.glob(os.path.join(args.input_path, '*.jpeg'))
    jpg_files = glob.glob(os.path.join(args.input_path, '*.jpg'))
    image_files = jpeg_files + jpg_files

    os.makedirs(args.output_path, exist_ok=True)

    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        output_file = os.path.basename(image_file)
        output_file = os.path.join(args.output_path, output_file)
        print(output_file, '{}/{}'.format(i, len(image_files)))
        cv2.imwrite(output_file, img_gray)


if __name__ == "__main__":
    main()

