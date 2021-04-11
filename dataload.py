from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.pose_logic_lib import normalize_relative_keypoints, preprocess_to_relative, \
    decode_pose_normalized_keypoints, decode_pose_normalized_keypoints_no_scaling

class a2kData(Dataset):
    def __init__(self, args, set_name, config):
        df = pd.read_csv(args.train_csv)
        if args.speaker != None:
            df = df[df['speaker'] == args.speaker]
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
        y = preprocess_to_relative(arr['pose'])
        y = normalize_relative_keypoints(y, row['speaker'])
        if "flatten" in self.config and self.config["flatten"]:
            y = y.flatten()
        else:
            y = np.swapaxes(y, 2, 1)
        return x, y

    def preprocess_x(self, x):
        """
        Zero pads audio file at the end to fixed size.

        Parameters
        ----------
        x : numpy.ndarray of shape (L,)

        Returns
        -------
        x : numpy.ndarray of shape (config.input_shape[1],)
        """
        if len(x) > self.config['input_shape'][1]:
            x = x[:self.config['input_shape'][1]]
        elif len(x) < self.config['input_shape'][1]:
            x = np.pad(x, [0, self.config['input_shape'][1] - len(x)],
                       mode='constant', constant_values=0)
        return x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        X : torch.tensor of shape (config.input_shape[1],)
        Y : torch.tensor of shape (64, 136)
        """
        row = self.df.iloc[idx]
        x_sample, y_sample = self.process_row(row)
        Y = torch.from_numpy(y_sample)
        X = torch.from_numpy(x_sample)
        return X, Y

# import pandas as pd
# df = pd.read_csv("train.csv")
AUDIO_SHAPE = 67267
configs = {
    "input_shape": [None, AUDIO_SHAPE],
    "audio_to_pose": {"num_keypoints": 136, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
    "audio_to_pose_inference": {"num_keypoints": 136, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
}

# path = r"C:\Users\sidhu\Desktop\train.csv"
mydata = a2kData(args, "train", configs["audio_to_pose"]).__getitem__([0])
