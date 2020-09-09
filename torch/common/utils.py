#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Miscellaneous utility functions."""
import os
import numpy as np
import cv2
from PIL import Image


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

