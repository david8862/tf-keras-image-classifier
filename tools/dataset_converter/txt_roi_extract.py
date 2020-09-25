#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import cv2
import argparse
from tqdm import tqdm

def get_ann(txt_file):
    '''loads the classes'''
    with open(txt_file) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Input path for the origin image', type=str, required=True)
    parser.add_argument('--output_path', help='Output path for the renamed image', type=str, required=True)
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(args.input_path, '*.txt'))

    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(total=len(txt_files), desc='working')
    for i, txt_file in enumerate(txt_files):
        base_name = os.path.basename(txt_file).split('.')[0]
        jpg_file = base_name + '.jpg'
        jpg_path = os.path.join('/mnt/nfs/Masked/VOC2007/JPEGImages/', jpg_file)
        #jpg_path = os.path.join('/mnt/nfs/b3day/', jpg_file)
        img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        height, width, channel = img.shape

        anno_list = get_ann(txt_file)
        for j, anno in enumerate(anno_list):
            line = anno.split()
            xmin,ymin,xmax,ymax,score = int(float(line[0])), int(float(line[1])),int(float(line[2])),int(float(line[3])),float(line[4])

            if xmax-xmin < 300:
                xmax += (300 - (xmax - xmin))//2
                xmin -= (300 - (xmax - xmin))//2
                xmax = min(width, xmax)
                xmin = max(0, xmin)

            if ymax-ymin < 300:
                ymax += (300 - (ymax - ymin))//2
                ymin -= (300 - (ymax - ymin))//2
                ymax = min(height, ymax)
                ymin = max(0, ymin)

            if (score > 0.8):
            #if (score > 0.6) and (xmax-xmin>50) and (ymax-ymin>50):
                area = img[ymin:ymax, xmin:xmax]
                output_img_name = os.path.join(args.output_path, '%s_%d.jpg'%(base_name, j))
                cv2.imwrite(output_img_name, area)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()

