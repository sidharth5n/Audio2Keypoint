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
        self.df = df[df['dataset'] == set_name]
        self.config = config
        self.to_tensor = transforms.ToTensor()

    def get_processor(self):
        processing_type = "audio_to_pose" #self.config["processor"]
        f = self.audio_pose_mel_spect
        if processing_type == 'audio_to_pose':
            d = decode_pose_normalized_keypoints
        elif processing_type == 'audio_to_pose_inference':
            d = decode_pose_normalized_keypoints_no_scaling
        else:
            raise ValueError("Wrong Processor")
        # return partial(f, self.config), d
        return f,d

    def audio_pose_mel_spect(self,row):
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
        # if torch.is_tensor(idx):
        # idx = idx.tolist()
        idx=[idx]
        process_row, decode_pose = self.get_processor()
        X, Y = [], []
        for i in idx:
            row = self.df.iloc[i]
            x_sample, y_sample = process_row(row)
            print("dataloader")
            y_sample=y_sample.T
            print(y_sample.shape)
            X.append(x_sample)
            Y.append(y_sample)
        # Y = self.to_tensor(Y)
        # X = self.to_tensor(X)
        X=np.array(X)
        Y=np.array(Y)
        Y = torch.from_numpy(Y)#.float() for item in Y
        
        X = torch.from_numpy(X)#.float() for item in X
        return X, Y
# AUDIO_SHAPE = 67267
# import pandas as pd
# my_dict={"audio_fn":["./Gestures/human/train/audio/id0004462OEFEevKvs00001-00:00:00.040000-00:00:05.360000.wav"],
# "dataset":["train"],"end":["0:00:02.560000"],"interval_id":["id037619diQL48epnM00019"],
# "pose_fn":["./Gestures/human/train/npz/id0004.npz"],
# "speaker":["human"],"start":["0:00:00.040000"],"video_fn":["id03761/9diQL48epnM/00019.mp4"]}
# df=pd.DataFrame.from_dict(my_dict)


# # ./Gestures/human/train/audio/id037619diQL48epnM00019-00:00:00.040000-00:00:18.600000.wav,train,0:00:02.560000,id037619diQL48epnM00019,
# # ./Gestures/human/train/npz/id037619diQL48epnM00019-0:00:00.040000-0:00:02.560000.npz,human,0:00:00.040000,id03761/9diQL48epnM/00019.mp4
# # df = pd.read_csv("train.csv")


# configs = {
#     "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
#     "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
# }
# mydata=a2kData(df,"train",configs["audio_to_pose"]).__getitem__([0])
# print(mydata[0].shape)
# print(mydata[1].shape)
# print(mydata[0].shape)



