#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import OrderedDict
import cv2

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_count = {}

def extract_roi(dataset_path, year, image_id, output_path, include_difficult):
    img_file_name = '%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id)
    # check if the image file exists
    if not os.path.exists(img_file_name):
        file_string = '%s/VOC%s/JPEGImages/%s.jpeg'%(dataset_realpath, year, image_id)
    if not os.path.exists(img_file_name):
        raise ValueError('image file for id: {} not exists'.format(image_id))

    img = cv2.imread(img_file_name, cv2.IMREAD_COLOR)

    xmin = int(img.shape[1]*0.35)
    ymin = int(img.shape[0]*0.7)
    xmax = int(img.shape[1]*0.65)
    ymax = int(img.shape[0]-1)
    # bypass invalid box
    #if (xmin >= xmax) or (ymin >= ymax):
        #continue

    area = img[ymin:ymax, xmin:xmax]
    output_img_name = os.path.join(output_path, '%s.jpg'%(image_id))
    cv2.imwrite(output_img_name, area)



def has_object(dataset_path, year, image_id, include_difficult):
    try:
        xml_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id))
    except:
        # bypass image if no annotation
        return False
    tree=ET.parse(xml_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        count = count +1
    return count != 0


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


parser = argparse.ArgumentParser(description='convert PascalVOC dataset annotation to txt annotation file')
parser.add_argument('--dataset_path', type=str, help='path to PascalVOC dataset, default=%(default)s', default=os.getcwd()+'/../../VOCdevkit')
parser.add_argument('--year', type=str, help='subset path of year (2007/2012), default will cover both', default=None)
parser.add_argument('--set', type=str, help='convert data set, default will cover train, val and test', default=None)
parser.add_argument('--output_path', type=str,  help='output path for generated annotation txt files, default=%(default)s', default='./')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions')
parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)
args = parser.parse_args()

# update class names
if args.classes_path:
    classes = get_classes(args.classes_path)

# get real path for dataset
dataset_realpath = os.path.realpath(args.dataset_path)

# create output path
os.makedirs(args.output_path, exist_ok=True)


# get specific sets to convert
if args.year is not None:
    sets = [item for item in sets if item[0] == args.year]
if args.set is not None:
    sets = [item for item in sets if item[1] == args.set]

for year, image_set in sets:
    # count class item number in each set
    class_count = OrderedDict([(item, 0) for item in classes])

    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(dataset_realpath, year, image_set)).read().strip().split()
    pbar = tqdm(total=len(image_ids), desc='Converting VOC%s_%s'%(year, image_set))
    for image_id in image_ids:
        file_string = '%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id)
        # check if the image file exists
        if not os.path.exists(file_string):
            file_string = '%s/VOC%s/JPEGImages/%s.jpeg'%(dataset_realpath, year, image_id)
        if not os.path.exists(file_string):
            raise ValueError('image file for id: {} not exists'.format(image_id))

        if has_object(dataset_realpath, year, image_id, args.include_difficult):
            extract_roi(dataset_realpath, year, image_id, args.output_path, args.include_difficult)

        pbar.update(1)
    pbar.close()

    # print out item number statistic
    print('\nDone for %s_%s.txt. classes number statistic'%(year, image_set))
    print('Image number: %d'%(len(image_ids)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))

