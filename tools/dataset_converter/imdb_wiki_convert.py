#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert IMDB-WIKI face image dataset
to imagenet format single face attribute (gender/age) classification dataset.

IMDB-WIKI face age and gender dataset:
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
here we just use the cropped face data
"""
import os, sys, argparse
import scipy.io
import numpy as np
import shutil
from datetime import datetime
from tqdm import tqdm

FACE_SCORE_THRESHOLD = 3
AGE_BOTTOM = 0
AGE_TOP = 100


def convert_gender_dataset(dataset_path, dataset, output_path):
    # get needed labels from mat struct
    image_names_array = dataset.full_path
    gender = dataset.gender
    face_score = dataset.face_score
    second_face_score = dataset.second_face_score

    # form data filter mask with related labels
    face_score_mask = face_score > FACE_SCORE_THRESHOLD
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender))
    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)

    # filter images
    image_names_array = image_names_array[mask]
    image_names = list(image_names_array)

    # filter gender labels
    gender_labels = gender[mask].tolist()

    # create output path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'female'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'male'), exist_ok=True)

    # move images to class path
    pbar = tqdm(total=len(image_names), desc='Convert gender dataset')
    for image_name, gender in zip(image_names, gender_labels):
        pbar.update(1)
        if gender == 0: #female face
            src_path = os.path.join(dataset_path, image_name)
            target_path = os.path.join(output_path, 'female', os.path.basename(image_name))
            #shutil.move(src_path, target_path)
            shutil.copy(src_path, target_path)
        elif gender == 1: #male face
            src_path = os.path.join(dataset_path, image_name)
            target_path = os.path.join(output_path, 'male', os.path.basename(image_name))
            #shutil.move(src_path, target_path)
            shutil.copy(src_path, target_path)
        else:
            raise ValueError('Unknown gender type:', gender)
    pbar.close()


def convert_age_dataset(dataset_path, dataset, output_path, num_age_classes):
    # get needed labels from mat struct
    image_names_array = dataset.full_path
    face_score = dataset.face_score
    second_face_score = dataset.second_face_score

    # calculate age label from 'photo_taken' and 'dob'
    photo_taken = dataset.photo_taken
    dob = np.asarray(list(map(lambda x: int(datetime.fromordinal(int(x)).strftime('%Y')), dataset.dob)))
    dob = np.asarray(list(map(lambda x: int(x), dob)))
    age = photo_taken - dob

    # form data filter mask with related labels
    face_score_mask = face_score > FACE_SCORE_THRESHOLD
    second_face_score_mask = np.isnan(second_face_score)
    mask = np.logical_and(face_score_mask, second_face_score_mask)

    # age range filter mask, only keep 0~100 years
    bottom_check = age > AGE_BOTTOM
    top_check = age < AGE_TOP
    age_mask = np.logical_and(bottom_check, top_check)
    mask = np.logical_and(mask, age_mask)

    # filter images
    image_names_array = image_names_array[mask]
    image_names = list(image_names_array)

    # filter age labels
    age_labels = age[mask].tolist()
    age_labels = [age_filter(age, num_age_classes) for age in age_labels]

    # create output path
    os.makedirs(output_path, exist_ok=True)
    for age_classes in range(num_age_classes):
        age_label_name = str(((AGE_TOP-AGE_BOTTOM)//num_age_classes) * (age_classes+1))
        os.makedirs(os.path.join(output_path, age_label_name), exist_ok=True)

    # move images to class path
    pbar = tqdm(total=len(image_names), desc='Convert age dataset')
    for image_name, age_label in zip(image_names, age_labels):
        pbar.update(1)
        age_label_name = str(((AGE_TOP-AGE_BOTTOM)//num_age_classes) * (age_label+1))
        src_path = os.path.join(dataset_path, image_name)
        target_path = os.path.join(output_path, age_label_name, os.path.basename(image_name))
        #shutil.move(src_path, target_path)
        shutil.copy(src_path, target_path)
    pbar.close()


def age_filter(age, num_age_classes):
    return int((age)/((AGE_TOP-AGE_BOTTOM)//num_age_classes))



def imdb_wiki_convert(dataset_path, dataset_file, dataset_name, output_path, select_attribute, num_age_classes):
    # load dataset file
    dataset = scipy.io.loadmat(dataset_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
    dataset = dataset[dataset_name]

    if select_attribute == 'gender':
        convert_gender_dataset(dataset_path, dataset, output_path)
    elif select_attribute == 'age':
        convert_age_dataset(dataset_path, dataset, output_path, num_age_classes)
    else:
        raise ValueError('Unsupported attribute type:', select_attribute)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert IMDB-WIKI face image dataset to imagenet format single face attribute (gender/age) classification dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='path to IMDB-WIKI dataset')
    parser.add_argument('--dataset_file', type=str, required=True, help='IMDB-WIKI dataset annotation .mat file')
    parser.add_argument('--dataset_name', type=str, required=False, default='imdb', choices=['imdb', 'wiki'], help='dataset name, default=%(default)s')
    parser.add_argument('--output_path', type=str, required=True,  help='output path')
    parser.add_argument('--select_attribute', type=str, required=False, default='gender', choices=['gender', 'age'], help='selected face attribute, default=%(default)s')
    parser.add_argument('--num_age_classes', type=int, required=False, default=10, help = "number of age groups for 0~100 years, default=%(default)s")

    args = parser.parse_args()

    imdb_wiki_convert(args.dataset_path, args.dataset_file, args.dataset_name, args.output_path, args.select_attribute, args.num_age_classes)



if __name__ == '__main__':
    main()

