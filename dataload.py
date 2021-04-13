from __future__ import print_function, division
import os
import threading
import numpy as np
from functools import partial
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

from common.pose_logic_lib import normalize_relative_keypoints, preprocess_to_relative, \
    decode_pose_normalized_keypoints, get_pose, decode_pose_normalized_keypoints_no_scaling


class a2kData(Dataset):
    def __init__(self, df, set_name, config):
        # Need to add condition for speaker if needed
        # see sid's master/dataload.py
        self.df = df[df['dataset'] == set_name]
        self.config = config
        self.process_row, self.decode_pose = self.get_processor()

    def get_processor(self):
        processing_type = self.config["processor"]
        f = self.audio_pose_mel_spect
        if processing_type == 'audio_to_pose':
            d = decode_pose_normalized_keypoints
        elif processing_type == 'audio_to_pose_inference':
            d = decode_pose_normalized_keypoints_no_scaling
        else:
            raise ValueError("Wrong Processor")
        return f, d

    def audio_pose_mel_spect(self, row):
        if "audio" in row:
            x = row["audio"]
        else:
            arr = np.load(row['pose_fn'])
            x = arr["audio"]
        x = self.preprocess_x(x)
        y = preprocess_to_relative(get_pose(arr))
        y = normalize_relative_keypoints(y, row['speaker'])
        if "flatten" in self.config and self.config["flatten"]:
            y = y.flatten()
        return x, y

    def preprocess_x(self, x):
        if len(x) > self.config["input_shape"][1]:
            x = x[:self.config['input_shape'][1]]
        elif len(x) < self.config['input_shape'][1]:
            x = np.pad(x, [0, self.config['input_shape'][1] - len(x)],
                       mode='constant', constant_values=0)
        return x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # idx = [idx]
        # process_row, decode_pose = self.get_processor()
        # X, Y = [], []
        # for i in idx:
        #     row = self.df.iloc[i]
        #     x_sample, y_sample = process_row(row)
        #     y_sample = y_sample.T
        #     X.append(x_sample)
        #     Y.append(y_sample)
        # Y = [torch.from_numpy(item).float() for item in Y]
        # X = [torch.from_numpy(item).float() for item in X]
        row = self.df.iloc[idx]
        x_sample, y_sample = self.process_row(row)
        Y = torch.from_numpy(y_sample)
        X = torch.from_numpy(x_sample)
        return X, Y
