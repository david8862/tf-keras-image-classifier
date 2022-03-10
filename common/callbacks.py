#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""custom model callbacks."""
import os, sys
import glob
from tensorflow.keras.callbacks import Callback


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep

    def on_epoch_end(self, epoch, logs=None):

        # filter out checkpoints
        checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')), reverse=False)

        # keep latest checkpoints
        for checkpoint in checkpoints[:-(self.max_keep)]:
            os.remove(checkpoint)

